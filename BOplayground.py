device = "cuda"
DEVICE = "cuda"
import os
import random
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.distributions.normal import Normal
import sympy

from code_construction.code_construction import CodeConstructor
from bayesian_optimization.objective_function import ObjectiveFunction
from bayesian_optimization.encoder import *
from bayesian_optimization.chaincomplexembedding import *
from bayesian_optimization.gp import *
from bayesian_optimization.bo import BO_on_QEC

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, SpectralMixtureKernel, ScaleKernel
import copy
from typing import Dict, Optional, Tuple, Any, List, Union
from dataclasses import dataclass, asdict


class HillClimbing:
    def __init__(
        self,
        next_points_num,
        gnp,
        acquisition,
        device="cuda",
        validator=None,
        method="bb",
        l=None,
        target_row_weight=None,
    ):

        self.next_points_num = int(next_points_num)
        self.gnp = gnp
        self.acquisition = acquisition
        self.device = device
        self.validator = (
            validator  # 例如 lambda z: code_constructor.construct(z).k != 0
        )
        self.l = l
        self.target_row_weight = target_row_weight
        match method:
            case "gb":
                self.mutate = self.swap_factors
            case "bb":
                self.mutate = self.flip_neighbours
            case _:
                raise ValueError("The method is not supported.")

    def flip_neighbours(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        d = x.numel()
        neigh_list = []
        for i in range(d):
            n = x.clone()
            # 0/1 翻转（注意 float32）
            n[i] = 1.0 - n[i]
            neigh_list.append(n)
        return torch.stack(neigh_list, dim=0)  # [d, d]

    def swap_factors(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()

        gx_mask = int(x[0].item())
        qa = int(x[1].item())
        qb = int(x[2].item())
        fs = [int(f.item()) for f in x[3:]]

        gx_bin = CodeConstructor.gx_mask_to_bin(gx_mask, fs, self.l)

        max_bound = self.l - (gx_bin.bit_length() - 1)

        neigh_list = []
        for i in range(max_bound):
            for j in range(i, max_bound):
                if i == j:
                    new_qa = qa ^ (1 << i)  # flips one factor of qa
                    new_qb = qb ^ (1 << i)
                else:
                    new_qa = qa ^ (1 << i) ^ (1 << j)  # flips two factors of qa
                    new_qb = qb ^ (1 << i) ^ (1 << j)

                a_weight = CodeConstructor.multiply_polynomials_mod_l(
                    gx_bin, new_qa, self.l
                ).bit_count()

                b_weight = CodeConstructor.multiply_polynomials_mod_l(
                    gx_bin, new_qb, self.l
                ).bit_count()

                if (
                    a_weight == self.target_row_weight
                ):  # checks if new a(x) preserves LDPC density
                    n1 = x.clone()
                    n1[1] = new_qa
                    neigh_list.append(n1)

                if (
                    b_weight == self.target_row_weight
                ):  # checks if new b(x) preserves LDPC density
                    n2 = x.clone()
                    n2[2] = new_qb
                    neigh_list.append(n2)

        if not neigh_list:
            return torch.empty((0, len(x)), dtype=x.dtype, device=x.device)

        return torch.stack(neigh_list, dim=0)

    @torch.no_grad()
    def __call__(self, gp) -> torch.Tensor:
        X0_np = self.gnp(self.next_points_num)  # [n, d], 0/1
        X0 = torch.tensor(X0_np, dtype=torch.float64, device=self.device)

        best_list = []
        for x in X0:  # x: [d]
            best_neighbor = x.clone()
            best_val = self.acquisition(best_neighbor.unsqueeze(0), gp).reshape(-1)[0]

            while True:
                nbrs = self.mutate(best_neighbor).to(self.device)
                acq_vals = self.acquisition(nbrs, gp).reshape(-1)  # [d]
                top_val, top_idx = torch.topk(acq_vals, k=1)
                top_val = top_val[0]
                top_neighbor = nbrs[top_idx[0]]

                if top_val.item() <= best_val.item():
                    break
                else:
                    best_neighbor = top_neighbor
                    best_val = top_val

            best_list.append(best_neighbor)

        cand = torch.stack(best_list, dim=0)  # [n, d], float64, 0/1

        if self.validator is not None:
            cand_np = cand.detach().cpu().numpy().astype(np.int64)
            for i in range(cand_np.shape[0]):
                if not self.validator(cand_np[i]):
                    tries = 0
                    while not self.validator(cand_np[i]):
                        tries += 1
                        cand_np[i] = self.gnp(1)
                        if tries > 1000:
                            break
            cand = torch.tensor(cand_np, dtype=torch.float64, device=self.device)

        return cand  # [n, d], float64(0/1), on device


class EIAcquisitionFunction:
    """
    Expected Improvement (EI) acquisition function for Bayesian Optimization.

    Designed for QEC-related objectives where the model is trained in the latent
    z-domain (log-transformed probability). Internally:
        - Converts GP posterior (μ_z, σ_z) → probability domain via LogStdNormalizer
        - Converts (μ_p, σ_p) → objective space using user-defined mapping pl_to_obj_fn
        - Computes standard EI = E[max(f(x) - f*, 0)]

    Parameters
    ----------
    pl_to_obj_fn : Callable
        Function mapping (X, μ_p, σ_p) to corresponding (μ_obj, σ_obj).
        It may return a tuple, list, or dict containing {'mean', 'std'} or {'mu', 'sigma'}.
    best_value : float, optional
        Current best observed objective value f*.
    jitter : float, default=1e-2
        Small positive offset to encourage exploration.
    eps : float, default=1e-9
        Numerical floor to stabilize variance/division.
    device : str, default="cpu"
        Computation device.
    normalizer : LogStdNormalizer, optional
        Optional external normalizer. If None, a new one is created internally.
    prob_lower, prob_upper : float
        Clamping bounds for probability space.
    log_eps : float
        Epsilon used in log(pl + eps) transform.
    """

    def __init__(
        self,
        pl_to_obj_fn,
        best_value=None,
        jitter=1e-2,
        eps=1e-9,
        device="cpu",
        normalizer=None,
        prob_lower=1e-12,
        prob_upper=1.0 - 1e-12,
        log_eps=1e-8,
    ):
        self.pl_to_obj_fn = pl_to_obj_fn
        self.best_value = best_value
        self.jitter = float(jitter)
        self.eps = float(eps)
        self.device = device
        self.prob_lower = float(prob_lower)
        self.prob_upper = float(prob_upper)
        self.log_eps = float(log_eps)

        # EI maintains its own normalizer; create one if not provided.
        self.normalizer = (
            normalizer
            if normalizer is not None
            else LogStdNormalizer(
                eps=self.log_eps,
                device=self.device,
                min_prob=self.prob_lower,
                max_prob=self.prob_upper,
            )
        )

    # ---------- Interface exposed to BO loop ----------
    def update_normalizer(self, pl_train: torch.Tensor):
        """Fit or update the internal normalizer using all observed pl values."""
        self.normalizer.fit(pl_train)

    def targets_from_pl(self, pl_train: torch.Tensor) -> torch.Tensor:
        """Map probability-domain pl → latent z-domain (for GP training targets)."""
        return self.normalizer.transform(pl_train)

    # ---------- Utility ----------
    def set_best_value(self, v):
        """Update the incumbent best objective value f*."""
        if torch.is_tensor(v):
            v = v.detach().item()
        self.best_value = float(v)

    def _take_mean_std(self, ret):
        """
        Parse the return of pl_to_obj_fn. It can be:
          - tuple/list: (mu, std)
          - dict: with keys ['mean', 'std'] or ['mu', 'sigma']
        """
        if isinstance(ret, (tuple, list)):
            return ret[0], ret[1]
        if isinstance(ret, dict):
            mu = ret.get("mean", ret.get("mu"))
            sd = ret.get("std", ret.get("sigma"))
            if mu is None or sd is None:
                raise RuntimeError(
                    "pl_to_obj_fn dict must contain mean/std (or mu/sigma)."
                )
            return mu, sd
        raise RuntimeError(f"Unsupported return type from pl_to_obj_fn: {type(ret)}")

    @torch.no_grad()
    def __call__(self, X, gp: torch.nn.Module):
        """
        Compute EI(x) for candidate inputs X under the current GP model.

        Steps:
          1) Compute GP posterior in latent z-domain: (μ_z, σ_z)
          2) Convert to probability domain using normalizer.inverse_mean_std()
          3) Map (μ_p, σ_p) → objective domain via pl_to_obj_fn
          4) Compute classical Expected Improvement:
                 EI(x) = (μ_f - f* - ξ) Φ(Z) + σ_f φ(Z),
             where Z = (μ_f - f* - ξ) / σ_f, Φ = CDF, φ = PDF of standard normal.
        """
        assert self.best_value is not None, (
            "best_value not set; call set_best_value() first."
        )

        # Ensure X is a proper tensor on the correct device
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float64)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        X = X.to(self.device, dtype=torch.float64)

        # Evaluation mode for GP
        gp.eval()
        if hasattr(gp, "likelihood") and gp.likelihood is not None:
            gp.likelihood.eval()

        # Posterior in latent z-domain
        with gpytorch.settings.fast_pred_var():
            if hasattr(gp, "posterior"):  # BoTorch-style API
                post = gp.posterior(X)
                mu_lat = post.mean.reshape(-1)
                var_lat = post.variance.reshape(-1)
            else:  # Pure GPyTorch model
                mvn = gp(X)
                mu_lat = mvn.mean.reshape(-1)
                var_lat = mvn.variance.reshape(-1)

        std_lat = var_lat.clamp_min(self.eps).sqrt()

        # Convert (μ_z, σ_z) → probability domain
        if not self.normalizer.is_fitted():
            raise RuntimeError(
                "EI normalizer is not fitted. Call update_normalizer(pl_train) first."
            )
        mu_pl, std_pl = self.normalizer.inverse_mean_std(mu_lat, std_lat)

        # Clamp probability and std for stability
        mu_pl = mu_pl.clamp(self.prob_lower, self.prob_upper)
        std_pl = std_pl.clamp_min(self.eps)

        # Map to objective domain
        mu_obj = torch.empty_like(mu_pl)
        std_obj = torch.empty_like(std_pl)
        for i in range(mu_pl.numel()):
            out = self.pl_to_obj_fn(X[i], mu_pl[i], std_pl[i])
            mu_i, sd_i = self._take_mean_std(out)
            mu_obj[i] = (
                mu_i
                if torch.is_tensor(mu_i)
                else torch.tensor(mu_i, device=self.device, dtype=torch.float32)
            )
            std_obj[i] = (
                sd_i
                if torch.is_tensor(sd_i)
                else torch.tensor(sd_i, device=self.device, dtype=torch.float32)
            )

        std_obj = std_obj.clamp_min(self.eps)

        # --- Classical Expected Improvement formula ---
        imp = mu_obj - float(self.best_value) - self.jitter
        Z = imp / std_obj
        normal = Normal(torch.zeros_like(mu_obj), torch.ones_like(std_obj))
        ei = imp * normal.cdf(Z) + std_obj * torch.exp(normal.log_prob(Z))
        return ei


TensorLike = Union[torch.Tensor, np.ndarray, float]


@dataclass
class LogStdStats:
    """Stores the fitted statistics (mean, std, eps) for log-domain normalization."""

    mean: float  # mean of log(pl + eps)
    scale: float  # std of log(pl + eps)
    eps: float  # epsilon used in log(pl + eps)


class LogStdNormalizer:
    """
    A normalizer that transforms probabilities `pl ∈ (0,1)` into a standardized latent space via:

        z = (log(pl + eps) - mean) / scale

    """

    def __init__(
        self,
        eps: float = 1e-8,
        min_prob: float = 1e-12,
        max_prob: float = 1.0 - 1e-12,
        device: str = "cpu",
    ):
        self.stats: LogStdStats = LogStdStats(mean=0.0, scale=1.0, eps=float(eps))
        self._fitted: bool = False
        self.min_prob = float(min_prob)
        self.max_prob = float(max_prob)
        self.device = device

    # -------------------- Basic utilities --------------------
    def _to_tensor(self, x: TensorLike, dtype=torch.float32) -> torch.Tensor:
        """Convert input (np.ndarray, float, or Tensor) to a torch.Tensor on the correct device."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=dtype)
        arr = np.asarray(x)
        return torch.tensor(arr, device=self.device, dtype=dtype)

    def _clamp_prob(self, pl: torch.Tensor) -> torch.Tensor:
        """Clamp probabilities to [min_prob, max_prob] for numerical stability."""
        return pl.clamp(self.min_prob, self.max_prob)

    # -------------------- Fitting and transforms --------------------
    @torch.no_grad()
    def fit(self, pl: TensorLike) -> "LogStdNormalizer":
        """
        Fit mean and scale parameters from all observed probabilities (in log-domain).
        """
        pl_t = self._to_tensor(pl, dtype=torch.float32).view(-1)
        pl_t = self._clamp_prob(pl_t)
        ylog = torch.log(pl_t + self.stats.eps)
        mean = ylog.mean().item()
        std = ylog.std(unbiased=False).item()  # population standard deviation

        # Avoid degenerate scaling (very small std can cause instability)
        if std < 1e-12:
            std = 1.0

        self.stats = LogStdStats(mean=float(mean), scale=float(std), eps=self.stats.eps)
        self._fitted = True
        return self

    @torch.no_grad()
    def transform(self, pl: TensorLike) -> torch.Tensor:
        """
        Transform from probability domain → standardized latent domain:
            z = (log(pl + eps) - mean) / scale
        """
        assert self._fitted, "Call fit(pl_train) before transform."
        pl_t = self._to_tensor(pl, dtype=torch.float32)
        pl_t = self._clamp_prob(pl_t)
        ylog = torch.log(pl_t + self.stats.eps)
        z = (ylog - self.stats.mean) / self.stats.scale
        return z

    @torch.no_grad()
    def inverse_transform(self, z: TensorLike) -> torch.Tensor:
        """
        Deterministic inverse transform:
            Given z, return point estimate of pl (not posterior expectation).
            pl = exp(z * scale + mean) - eps
        """
        assert self._fitted, "Call fit(pl_train) before inverse_transform."
        z_t = self._to_tensor(z, dtype=torch.float32)
        logpl = z_t * self.stats.scale + self.stats.mean
        pl = torch.exp(logpl) - self.stats.eps
        return self._clamp_prob(pl)

    # -------------------- Posterior inverse transform (used in EI acquisition) --------------------
    @torch.no_grad()
    def inverse_mean_std(
        self, mu_z: TensorLike, std_z: TensorLike
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If z ~ N(mu_z, std_z^2), then log(pl + eps) ~ N(mu, std^2), where:
            mu  = mu_z * scale + mean
            std = |scale| * std_z

        Using log-normal moment formulas:
            E[pl + eps] = exp(mu + 0.5 * std^2)
            Var(pl + eps) = (exp(std^2) - 1) * exp(2mu + std^2)

        Returns:
            (mean_pl, std_pl): expected mean and std of pl in probability domain.
        """
        assert self._fitted, "Call fit(pl_train) before inverse_mean_std."
        mu_z_t = self._to_tensor(mu_z, dtype=torch.float32)
        std_z_t = self._to_tensor(std_z, dtype=torch.float32).abs()

        mu = mu_z_t * self.stats.scale + self.stats.mean
        std = std_z_t * abs(self.stats.scale)

        # E[pl + eps] and Var(pl + eps)
        exp_half_var = torch.exp(0.5 * std**2)
        mean_pl_plus = torch.exp(mu) * exp_half_var
        var_pl_plus = (torch.exp(std**2) - 1.0) * torch.exp(2.0 * mu + std**2)

        mean_pl = mean_pl_plus - self.stats.eps
        std_pl = var_pl_plus.clamp_min(1e-30).sqrt()

        # Clamp mean within valid probability range
        mean_pl = self._clamp_prob(mean_pl)
        return mean_pl, std_pl

    # -------------------- State handling and serialization --------------------
    def is_fitted(self) -> bool:
        """Return whether the normalizer has been fitted."""
        return self._fitted

    def get_stats(self) -> LogStdStats:
        """Return the current (mean, scale, eps) statistics."""
        return self.stats

    def set_eps(self, eps: float):
        """Update epsilon used in log(pl + eps)."""
        self.stats.eps = float(eps)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize current state to a Python dict."""
        return {
            "stats": asdict(self.stats),
            "fitted": self._fitted,
            "min_prob": self.min_prob,
            "max_prob": self.max_prob,
            "device": self.device,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore normalizer state from a dict."""
        s = state["stats"]
        self.stats = LogStdStats(
            mean=float(s["mean"]), scale=float(s["scale"]), eps=float(s["eps"])
        )
        self._fitted = bool(state.get("fitted", True))
        self.min_prob = float(state.get("min_prob", self.min_prob))
        self.max_prob = float(state.get("max_prob", self.max_prob))
        self.device = state.get("device", self.device)


class GPTrainer:
    """
    Trainer for an ExactGP with a fixed structure (e.g., ScaleKernel ∘ Matern).

    Public methods:
      - update_data(X, z)            : Hot-update the full training set
                                       (z is the normalized log(pl)).
      - on_normalizer_change(old,new): Rescale outputscale/noise when the scaler std changes.
      - maybe_freeze_unfreeze(round): Optionally freeze lengthscale in early rounds.
      - train_one_round()            : One training round (scheduler/early stop/grad clipping/priors/constraints).
    """

    def __init__(
        self,
        model: gpytorch.models.ExactGP,
        device: str = "cpu",
        training_iter: int = 80,
        lr: Optional[
            Dict[str, float]
        ] = None,  # {'embed':4e-4,'mean':1e-3,'kernel':1e-3,'like':2e-2}
        weight_decay: float = 1e-4,
        max_grad_norm: float = 2.0,
        optimizer_type: str = "adamw",
        recreate_optimizer_each_round: bool = True,
        scheduler_cfg: Optional[
            Dict[str, Any]
        ] = None,  # {'factor':0.5,'patience':5,'min_lr':1e-6}
        early_stopping: Optional[Dict[str, Any]] = None,  # {'patience':10,'tol':1e-4}
        use_priors: bool = True,
        priors_cfg: Optional[Dict[str, Dict[str, float]]] = None,
        noise_floor: float = 1e-3,  # lower bound (in z-domain) for likelihood noise
        lengthscale_bounds: Optional[Tuple[float, float]] = None,
        warm_start: bool = True,
        rescale_on_scaler_change: bool = True,  # alpha = old_std / new_std
        carry_optimizer_state: bool = False,
        freeze_cfg: Optional[Dict[str, Any]] = None,  # {'lengthscale_until_round': 2}
        verbose: bool = True,
        log_every: Optional[int] = None,
        save_best_state: bool = True,
    ):
        self.model = model.to(device)
        self.device = device

        self.training_iter = int(training_iter)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer_type = optimizer_type.lower()
        self.recreate_optimizer_each_round = bool(recreate_optimizer_each_round)

        self.scheduler_cfg = scheduler_cfg or {
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6,
        }
        self.early_stopping = early_stopping or {"patience": 10, "tol": 1e-4}

        self.use_priors = bool(use_priors)
        self.priors_cfg = priors_cfg or {
            "lengthscale": {"type": "lognormal", "loc": 0.0, "scale": 0.5},
            "outputscale": {"type": "lognormal", "loc": 0.0, "scale": 0.5},
            "noise": {"type": "lognormal", "loc": -4.0, "scale": 0.5},
        }
        self.noise_floor = float(noise_floor)
        self.lengthscale_bounds = lengthscale_bounds

        self.warm_start = bool(warm_start)
        self.rescale_on_scaler_change = bool(rescale_on_scaler_change)
        self.carry_optimizer_state = bool(carry_optimizer_state)

        self.freeze_cfg = freeze_cfg or {"lengthscale_until_round": 0}

        self.verbose = bool(verbose)
        self.log_every = log_every
        self.save_best_state = bool(save_best_state)
        self.lr = lr or {"embed": 4e-4, "mean": 1e-3, "kernel": 1e-3, "like": 2e-2}

        # ---- State ----
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
        self._best_state: Optional[Dict[str, Dict[str, Any]]] = None
        self._history: List[float] = []
        self._round_idx: int = 0
        self._last_scaler_std: Optional[float] = None

        # Priors (bound to constrained param names) & constraints (on raw_* parameters)
        if self.use_priors:
            self._register_priors_safely()
        self._register_constraints_safely()

    # ================= Public API =================
    def update_data(self, X: torch.Tensor, z: torch.Tensor):
        X = X.to(self.device).float().contiguous()
        z = z.to(self.device).float().view(-1).contiguous()
        # Some ExactGP versions use keyworded `set_train_data`, others positional.
        try:
            self.model.set_train_data(inputs=X, targets=z, strict=False)
        except TypeError:
            self.model.set_train_data(X, z, strict=False)

    def on_normalizer_change(self, old_std: Optional[float], new_std: Optional[float]):
        """
        When the external scaler's standard deviation changes, rescale the model's
        outputscale and noise by alpha^2, where alpha = old_std / new_std.
        """
        if not self.rescale_on_scaler_change:
            self._last_scaler_std = new_std
            return
        if old_std is None or new_std is None or new_std <= 0 or old_std <= 0:
            self._last_scaler_std = new_std
            return
        alpha = float(old_std / new_std)
        with torch.no_grad():
            if hasattr(self.model, "covar_module") and hasattr(
                self.model.covar_module, "outputscale"
            ):
                self.model.covar_module.outputscale.mul_(alpha**2)
            try:
                self.model.likelihood.noise.mul_(alpha**2)
            except Exception:
                # Fallback: attempt to read/set noise through likelihood noise_covar
                try:
                    noise = self._get_noise_value()
                    self._set_noise_value(noise * (alpha**2))
                except Exception:
                    pass
        self._last_scaler_std = new_std

    def maybe_freeze_unfreeze(self, round_idx: int):
        """
        Optionally freeze base kernel lengthscale parameters for early rounds.
        """
        self._round_idx = int(round_idx)
        until = int(self.freeze_cfg.get("lengthscale_until_round", 0))
        freeze = round_idx <= until
        base_kernel = getattr(
            getattr(self.model, "covar_module", None), "base_kernel", None
        )
        if base_kernel is not None:
            for p in base_kernel.parameters():
                p.requires_grad_(not freeze)

    def train_one_round(self) -> Dict[str, Any]:
        """
        Run one training round with:
          - optimizer setup (optionally recreated per round),
          - ReduceLROnPlateau scheduler,
          - early stopping,
          - gradient clipping,
          - best-state checkpointing (optional).
        """
        model = self.model
        model.train()
        model.likelihood.train()

        if (
            self._optimizer is None
            or self.recreate_optimizer_each_round
            or not self.carry_optimizer_state
        ):
            self._optimizer = self._build_optimizer()
            self._scheduler = self._build_scheduler(self._optimizer)
        elif self._scheduler is None:
            self._scheduler = self._build_scheduler(self._optimizer)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        best_loss = float("inf")
        best_state = None
        no_improve = 0
        patience = int(self.early_stopping.get("patience", 10))
        tol = float(self.early_stopping.get("tol", 1e-4))
        self._history = []

        train_x, train_y = model.train_inputs[0], model.train_targets

        for it in range(1, self.training_iter + 1):
            self._optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()

            if self.max_grad_norm and self.max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
            self._optimizer.step()

            cur = float(loss.item())
            self._history.append(cur)

            if self._scheduler is not None:
                self._scheduler.step(cur)

            rel_impr = (best_loss - cur) / (abs(best_loss) + 1e-9)
            if cur + 1e-6 < best_loss:
                best_loss = cur
                no_improve = 0
                if self.save_best_state:
                    best_state = {
                        "model": copy.deepcopy(model.state_dict()),
                        "likelihood": copy.deepcopy(model.likelihood.state_dict()),
                    }
            else:
                no_improve += 1

            if self.verbose and (
                self.log_every is None
                or it % max(1, self.log_every) == 0
                or it == self.training_iter
            ):
                ls_val = self._safe_get_lengthscale()
                out_v = self._safe_get_outputscale()
                nz_v = self._safe_get_noise()
                print(
                    f"[round {self._round_idx:>3d} | {it:>4d}/{self.training_iter}] "
                    f"nll={cur:.4f}  len={ls_val}  out={out_v:.3e}  noise={nz_v:.3e}"
                )

            if no_improve >= patience and rel_impr < tol:
                if self.verbose:
                    print(
                        f"[round {self._round_idx:>3d}] early stop @ {it}, best nll={best_loss:.4f}"
                    )
                break

        if self.save_best_state and best_state is not None:
            model.load_state_dict(best_state["model"])
            model.likelihood.load_state_dict(best_state["likelihood"])
            self._best_state = best_state

        return {
            "final_nll": float(self._history[-1]),
            "best_nll": float(best_loss),
            "history": [float(v) for v in self._history],
            "lengthscale": self._safe_get_lengthscale(),
            "outputscale": self._safe_get_outputscale(),
            "noise": self._safe_get_noise(),
            "round_idx": self._round_idx,
            "steps": len(self._history),
        }

    # ================= Internal Utilities =================
    def _build_optimizer(self) -> torch.optim.Optimizer:
        # Group parameters by module for separate learning rates.
        params_embed = list(
            getattr(self.model, "embed", torch.nn.Module()).parameters()
        )
        params_kernel = list(
            getattr(self.model, "covar_module", torch.nn.Module()).parameters()
        )
        params_like = list(self.model.likelihood.parameters())
        params_mean = list(
            getattr(self.model, "mean_module", torch.nn.Module()).parameters()
        )

        lr = self.lr
        # learning rate
        groups = [
            {
                "params": params_embed,
                "lr": lr.get("embed", 4e-4),
                "weight_decay": self.weight_decay,
            },
            {
                "params": params_mean,
                "lr": lr.get("mean", 1e-3),
                "weight_decay": self.weight_decay,
            },
            {
                "params": params_kernel,
                "lr": lr.get("kernel", 1e-3),
                "weight_decay": self.weight_decay,
            },
            {
                "params": params_like,
                "lr": lr.get("like", 2e-2),
                "weight_decay": self.weight_decay / 5,
            },
        ]
        if self.optimizer_type == "adam":
            return torch.optim.Adam(groups)
        return torch.optim.AdamW(groups)

    def _build_scheduler(self, optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        cfg = self.scheduler_cfg
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 5)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
        )

    def _register_priors_safely(self):
        """Attach priors to the **constrained** parameter names (lengthscale/outputscale/noise)."""
        from gpytorch.priors import LogNormalPrior

        # lengthscale prior
        try:
            bk = self.model.covar_module.base_kernel
            loc = float(self.priors_cfg["lengthscale"]["loc"])
            scale = float(self.priors_cfg["lengthscale"]["scale"])
            bk.register_prior(
                "lengthscale_prior", LogNormalPrior(loc, scale), "lengthscale"
            )
        except Exception:
            pass
        # outputscale prior
        try:
            loc = float(self.priors_cfg["outputscale"]["loc"])
            scale = float(self.priors_cfg["outputscale"]["scale"])
            self.model.covar_module.register_prior(
                "outputscale_prior", LogNormalPrior(loc, scale), "outputscale"
            )
        except Exception:
            pass
        # noise prior
        try:
            loc = float(self.priors_cfg["noise"]["loc"])
            scale = float(self.priors_cfg["noise"]["scale"])
            self.model.likelihood.register_prior(
                "noise_prior", LogNormalPrior(loc, scale), "noise"
            )
        except Exception:
            pass

    def _register_constraints_safely(self):
        # Noise lower bound (constraint on raw_noise)
        try:
            self.model.likelihood.noise_covar.register_constraint(
                "raw_noise",
                gpytorch.constraints.GreaterThan(
                    torch.tensor(self.noise_floor, device=self.device)
                ),
            )
        except Exception:
            pass
        # Optional lengthscale bounds (constraint on raw_lengthscale)
        if self.lengthscale_bounds is not None:
            lo, hi = self.lengthscale_bounds
            try:
                self.model.covar_module.base_kernel.register_constraint(
                    "raw_lengthscale", gpytorch.constraints.Interval(lo, hi)
                )
            except Exception:
                pass

    # ---- Safe getters/setters ----
    def _safe_get_lengthscale(self) -> Any:
        try:
            v = self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            return v.tolist()
        except Exception:
            return "n/a"

    def _safe_get_outputscale(self) -> float:
        try:
            return float(self.model.covar_module.outputscale.detach().cpu().item())
        except Exception:
            return float("nan")

    def _get_noise_value(self) -> float:
        try:
            return float(self.model.likelihood.noise.detach().cpu().item())
        except Exception:
            try:
                return float(
                    self.model.likelihood.noise_covar.noise.detach().cpu().item()
                )
            except Exception:
                return float("nan")

    def _set_noise_value(self, v: float):
        with torch.no_grad():
            try:
                self.model.likelihood.noise.copy_(torch.tensor(v, device=self.device))
            except Exception:
                try:
                    self.model.likelihood.noise_covar.noise.copy_(
                        torch.tensor(v, device=self.device)
                    )
                except Exception:
                    pass

    def _safe_get_noise(self) -> float:
        return self._get_noise_value()


class E:
    def __init__(self, code_constructor, views_info):
        self.code_constructor = code_constructor
        self.encoder = CSSEncoder(views_info, mode="relations")

    def encode_single(self, x):
        return self.encoder.encode(
            self.code_constructor.construct(np.array(x).astype(int))
        )

    def encode(self, x):
        # x: B views
        return [self.encode_single(i) for i in x]


views_info = [
    {
        "name": "decode",
        "partite_classes": ["SZ", "DQ", "SX"],
        "relations": ["SZ_DQ", "DQ_SZ", "DQ_SX", "SX_DQ"],
        "weight_mode": "count",
        "log1p": True,
    },
    {
        "name": "xlogic",
        "partite_classes": ["DQ", "SX", "LX"],
        "relations": ["DQ_SX", "SX_DQ", "LX_DQ", "DQ_LX"],
        "weight_mode": "count",
        "log1p": True,
    },
    {
        "name": "zlogic",
        "partite_classes": ["SZ", "DQ", "LZ"],
        "relations": ["SZ_DQ", "DQ_SZ", "LZ_DQ", "DQ_LZ"],
        "weight_mode": "count",
        "log1p": True,
    },
]


def get_model_(
    X, y, kernel_type="ard_rbf", mean_type="linear", mean_input=64, embed_dim=128
):

    encoder = E(code_constructor, views_info)

    # NN embedder
    embedding = ChainComplexEmbedder(
        views_info=views_info,
        d_model=embed_dim,
        num_layers=4,
        view_aggr="sum",
        num_bases=4,
        norm="sym",
        residual=True,
        self_loop=False,
        dropout=0.1,
    ).to(device)

    # kernel
    if kernel_type == "ard_rbf":
        base = RBFKernel(ard_num_dims=embed_dim)
        kernel = ScaleKernel(base)
    elif kernel_type == "matern":
        base = MaternKernel(nu=1.5, ard_num_dims=embed_dim)
        kernel = ScaleKernel(base)
    elif kernel_type == "spectral_mixture":
        kernel = SpectralMixtureKernel(num_mixtures=4, ard_num_dims=embed_dim)
    elif kernel_type == "rbf_plus_periodic":
        from gpytorch.kernels import PeriodicKernel

        rbf = ScaleKernel(RBFKernel(ard_num_dims=embed_dim))
        periodic = ScaleKernel(PeriodicKernel(ard_num_dims=embed_dim))
        kernel = rbf + periodic
    else:
        kernel = ScaleKernel(RBFKernel())

    # likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    # train_x/y to device, y -> 1D for ExactGP
    train_x = X.to(device).float()
    train_y = y.to(device).float().view(-1)

    gp = GaussianProcess_QEC(
        train_x,
        train_y,
        likelihood=likelihood,
        kernel=kernel,
        encoder=encoder.encode,
        embed=embedding,
        mean=mean_type,
        mean_input=mean_input,
    ).to(device)

    return gp


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Get_new_points_function:
    def __init__(
        self,
        method="qc-ldpc-hgp",
        code_constructor=None,
        encode="None",
        density=None,
        desired_k=None,
        gx_mask=None,
    ):
        self.method = method
        self.code_constructor = code_constructor
        self.encode = encode
        self.density = density
        self.init = False
        self.desired_k = desired_k
        self.gx_mask = gx_mask
        self.factors = None

    def get_new_points_function(self, number):
        if self.method == "qc-ldpc-hgp":
            new_points = self.get_new_points_HGP(number)
        elif self.method == "gb":
            new_points = self.get_new_gb_vector(number)
        elif self.method == "bb":
            new_points = self.get_new_bb_vector(number)
        return new_points

    def get_new_points_HGP(self, number):
        return np.random.randint(
            0,
            self.hyperparameters["m"] + 1,
            (number, self.hyperparameters["p"] * self.hyperparameters["q"]),
        )

    def _get_irreducible_factors(self, l) -> list[int]:
        x = sympy.Symbol("x")
        poly = x**l - 1
        _, factors_counts = sympy.factor_list(poly, domain=sympy.GF(2))

        int_factors = []
        for factor, count in factors_counts:
            coeffs = sympy.Poly(factor, x).all_coeffs()
            binary_coeffs = [int(c) % 2 for c in coeffs][::-1]

            # convert binary array to int
            factor_int = sum([bit * (1 << i) for i, bit in enumerate(binary_coeffs)])

            for _ in range(count):
                int_factors.append(factor_int)

        return int_factors

    def _get_possible_gx_bitmasks(self, factors):
        if self.desired_k is None:
            raise ValueError(
                "A desired value of k is required for GB codes if g(x) bitmask not provided."
            )

        target_degree = self.desired_k // 2
        no_factors = len(factors)

        degrees = [f.bit_length() - 1 for f in factors]
        valid_bitmasks = []

        for i in range(1, 1 << no_factors):  # iterate through all possible bitmasks
            degree = 0
            for j in range(no_factors):
                if (i >> j) & 1:
                    degree += degrees[j]
            if degree == target_degree:
                valid_bitmasks.append(i)

        return valid_bitmasks

    def set_gx_mask(self):
        l = self.code_constructor.para_dict["l"]
        self.factors = self._get_irreducible_factors(l)
        if self.gx_mask is not None:
            if not (self.gx_mask > 0 and self.gx_mask < (1 << len(self.factors))):
                raise ValueError(
                    f"Provided g(x) bitmask {self.gx_mask} is not valid: must be int in range [1, {(1 << len(self.factors)) - 1}] for l={l}"
                )
        else:
            possible_gx_masks = self._get_possible_gx_bitmasks(self.factors)
            self.gx_mask = random.choice(possible_gx_masks)
            print(f"Randomly selected g(x) bitmask = {self.gx_mask}")

    def get_new_gb_vector(self, number):
        results = []
        l = self.code_constructor.para_dict["l"]

        gx_bin = CodeConstructor.gx_mask_to_bin(self.gx_mask, self.factors, l)

        max_bound = 2 ** (l - (gx_bin.bit_length() - 1))

        while number > 0:
            batch_guesses = np.random.randint(1, max_bound, size=10_000)
            qa_found = False

            for qa in batch_guesses:
                qa = qa.item()

                # qa = np.random.randint(1, max_bound)
                weight_a = CodeConstructor.multiply_polynomials_mod_l(
                    gx_bin, qa, l
                ).bit_count()
                # if weight_a != self.density:
                #     continue
                if weight_a == self.density:
                    qa_found = True
                    break

            if qa_found:
                qb_found = False
                while not qb_found:
                    batch_guesses = np.random.randint(1, max_bound, size=10_000)

                    for qb in batch_guesses:
                        qb = qb.item()
                        weight_b = CodeConstructor.multiply_polynomials_mod_l(
                            gx_bin, qb, l
                        ).bit_count()
                        if weight_b == self.density:
                            qb_found = True
                            break

                parameters = [self.gx_mask, qa, qb] + self.factors

                results.append(parameters)
                number -= 1

        return np.array(results)

    def get_new_bb_vector(self, number):
        results = []
        l = self.code_constructor.para_dict["l"]
        g = self.code_constructor.para_dict["g"]
        if self.init == False and l == 12 and g == 999:
            print("best known bb code added to initial points")
            self.init = True

            a = np.zeros((l + g - 1) * 2)
            a[3] = 1
            a[11 + 1] = 1
            a[11 + 2] = 1
            a[17 + 1] = 1
            a[17 + 2] = 1
            a[17 + 11 + 3] = 1
            results.append(a)

        while number > 0:
            new_point = np.random.randint(0, 2, size=(l + g - 1) * 2)
            c = self.code_constructor.construct(new_point)
            if c.k == 0:
                continue
            else:
                results.append(new_point)
                number -= 1
        return np.array(results)


if __name__ == "__main__":
    import pickle
    import sys

    # default values
    code_class = "bb"
    density = 4
    seed = 42
    dataset_index = 0
    lambda_ = 1
    desired_k = None
    gx_mask = 1

    args = sys.argv[1:]
    if args[0].isalpha():
        code_class = args.pop(0)

    if len(args) >= 3:
        seed = int(args[1])
        dataset_index = int(args[2])
    if len(args) >= 4:
        lambda_ = float(args[3])
    if len(args) >= 5:
        density = int(args[4])
    if len(args) >= 6:
        if args[5][:2] == "k=":
            desired_k = int(args[5][2:])
        elif args[5][:2] == "g=":
            gx_mask = int(args[5][2:])

    set_all_seeds(seed)
    # l = 12
    l = 7
    g = 6  # g here is m in Bravyi et al's paper
    print(
        f"(l,g)=({l},{g}), dataset_index = {dataset_index}, seed={seed}, lambda = {lambda_}"
    )

    para_dict = {"l": l, "g": g}

    code_constructor = CodeConstructor(method=code_class, para_dict=para_dict)
    # define objective function
    pp = 0.05
    Obj_Func = ObjectiveFunction(
        code_constructor, lambda_=lambda_, pp=pp, decoder_param={"trail": 10_000}
    )
    obj_func = Obj_Func.forward
    pl_to_obj = Obj_Func.pl_to_obj_with_std
    # method of sampling new points
    gnp_obj = Get_new_points_function(
        method=code_class,
        code_constructor=code_constructor,
        density=density,
        desired_k=desired_k,
        gx_mask=gx_mask,
    )
    gnp = gnp_obj.get_new_points_function
    # initial points:
    # init_num = 20
    # X_init = gnp(init_num)
    # y_init = []
    # pl_init = []
    # for x in X_init:
    #     y,pl = obj_func(x)
    #     y_init.append(y)
    #     pl_init.append(pl)
    if code_class == "gb":
        gnp_obj.set_gx_mask()
        init_data_file = f"./data/BO_initial_points/GB_BO_initial_points_{dataset_index}_{lambda_}_{density}_{l}_{gnp_obj.gx_mask}.pkl"
    else:  # bb
        if l == 6 and g == 3:
            if lambda_ == 1:
                init_data_file = f"./data/BO_initial_points/BO_initial_points_{dataset_index}_1.0_63.pkl"
            else:
                init_data_file = f"./data/BO_initial_points/BO_initial_points_{dataset_index}_{lambda_}_63.pkl"

        else:
            if lambda_ == 1:
                init_data_file = f"./data/BO_initial_points/BO_initial_points_{dataset_index}_1.0.pkl"
            else:
                init_data_file = f"./data/BO_initial_points/BO_initial_points_{dataset_index}_{lambda_}.pkl"
        # file with 63 suffix has (l,m)=(6,3). Otherwise (l,m)=(12,6)

    if not os.path.exists(init_data_file):
        # no starting codes exist, so generate some:
        init_num = 20
        print(f"Generating and evaluating {init_num} initial {code_class} codes...")
        X_init = gnp(init_num)
        y_init = []
        pl_init = []
        for i, x in enumerate(X_init):
            y, pl = obj_func(x)
            y_init.append(y)
            pl_init.append(pl)
            print(f"Evaluated initial code {i}/{init_num} (score={y:.4f})")

        with open(init_data_file, "wb") as f:
            pickle.dump({"X": X_init, "y": y_init, "pl": pl_init}, f)
            print(f"Saved initial points to {init_data_file}")

    print(f"Loading initial codes from {init_data_file}")
    with open(init_data_file, "rb") as f:
        data = pickle.load(f)
        X_init = data["X"]
        y_init = data["y"]
        pl_init = data["pl"]

    X_init = torch.tensor(X_init, dtype=torch.float64)
    X_init.to(DEVICE)  # think this line should be X_init = X_init.to(DEVICE)
    y_init = torch.tensor(y_init, dtype=torch.float32)
    y_init.to(DEVICE)
    pl_init = torch.tensor(pl_init, dtype=torch.float32)
    pl_init.to(DEVICE)
    # get gp model
    model = get_model_(
        X_init,
        pl_init,
        kernel_type="matern",
        mean_type="linear",
        mean_input=128,
    )
    # gp trainer
    trainer = GPTrainer(
        model=model,
        device=DEVICE,
        # --- Training and regularization ---
        training_iter=80,
        lr={"embed": 4e-4, "mean": 1e-3, "kernel": 1e-3, "like": 2e-2},
        weight_decay=1e-4,
        max_grad_norm=2.0,
        optimizer_type="adamw",
        recreate_optimizer_each_round=True,
        # --- Scheduler and early stopping ---
        scheduler_cfg={"factor": 0.5, "patience": 5, "min_lr": 1e-6},
        early_stopping={"patience": 10, "tol": 1e-4},
        # --- Priors and constraints (helpful for small datasets) ---
        use_priors=True,
        noise_floor=1e-3,  # Lower bound for z-domain noise to avoid overfitting
        lengthscale_bounds=None,  # If X is normalized to [0,1], one may use (1e-2, 10.)
        # --- Warm start and scaler rescaling ---
        warm_start=True,
        rescale_on_scaler_change=True,  # Recommended: True
        carry_optimizer_state=False,
        # --- Freeze base kernel lengthscale in early rounds (stabilizes small-data regime) ---
        freeze_cfg={"lengthscale_until_round": 2},
        verbose=True,
        log_every=8,
        save_best_state=True,
    )
    # acquisition function:
    acq = EIAcquisitionFunction(
        pl_to_obj_fn=pl_to_obj,
        best_value=None,
        jitter=1e-2,
        eps=1e-9,
        device=DEVICE,
        normalizer=None,
        prob_lower=1e-12,
        prob_upper=1.0 - 1e-12,
        log_eps=1e-8,
    )

    # hill climbing:
    def code_validator(candidate_np):
        return code_constructor.construct(candidate_np).k != 0

    next_points_num = 4

    hc = HillClimbing(
        next_points_num=next_points_num,  # the candidate number
        gnp=gnp,  # get new points function
        acquisition=acq,
        device=DEVICE,
        validator=code_validator,
        method=code_class,
        l=l,
        target_row_weight=density,
    )

    # assemble BO
    bo_iterations = 50
    bo = BO_on_QEC(
        gp=model,
        gp_trainer=trainer,
        acquisition_function=acq,
        suggest_next=hc,
        objective_function=obj_func,
        initial_X=X_init,
        initial_pl=pl_init,
        initial_y=y_init,
        BO_iterations=bo_iterations,
        description=f"{code_class.upper()}-BO (GP+EI+HC)",
        device=DEVICE,
        pretrain=True,
    )
    best_x, best_y, evaluation_history = bo.run()

    # The best-so-far results of bo (including initial points)
    flat = [v for row in evaluation_history for v in row]
    y_init_list = [i.item() for i in y_init]
    flat = y_init_list + flat

    results_file = init_data_file.replace("BO_initial_points", "BO_results")

    with open(results_file, "wb") as f:
        results = {"best_x": best_x, "best_y": best_y, "evaluation_history": flat}
        pickle.dump(results, f)

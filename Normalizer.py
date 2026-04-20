from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

import numpy as np
import torch

from typing import Dict, Tuple, Any, Union


@dataclass
class StdStats:
    mean: float
    scale: float


@dataclass
class LogStdStats(StdStats):
    eps: float


TensorLike = Union[torch.Tensor, np.ndarray, float]


class Normalizer(ABC):
    def __init__(
        self,
        device: str = "cpu",
    ):
        self._fitted: bool = False
        self.device = device

    def _to_tensor(self, x: TensorLike, dtype=torch.float32) -> torch.Tensor:
        """Convert input (np.ndarray, float, or Tensor) to a torch.Tensor on the correct device."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=dtype)
        arr = np.asarray(x)
        return torch.tensor(arr, device=self.device, dtype=dtype)

    @abstractmethod
    def fit(self, y: TensorLike) -> Normalizer:
        pass

    @abstractmethod
    def transform(self, y: TensorLike) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse_transform(self, z: TensorLike) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse_mean_std(
        self, mu_z: TensorLike, std_z: TensorLike
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def is_fitted(self) -> bool:
        """Return whether the normalizer has been fitted."""
        return self._fitted

    def get_stats(self) -> LogStdStats:
        """Return the current (mean, scale, eps) statistics."""
        return self.stats

    def state_dict(self) -> Dict[str, Any]:
        """Serialize current state to a Python dict."""
        pass

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore normalizer state from a dict."""
        pass


class LogStdNormalizer(Normalizer):
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
        super().__init__(device)
        self.stats: LogStdStats = LogStdStats(mean=0.0, scale=1.0, eps=float(eps))
        self.min_prob = float(min_prob)
        self.max_prob = float(max_prob)

    # -------------------- Basic utilities --------------------

    def _clamp_prob(self, pl: torch.Tensor) -> torch.Tensor:
        """Clamp probabilities to [min_prob, max_prob] for numerical stability."""
        return pl.clamp(self.min_prob, self.max_prob)

    # -------------------- Fitting and transforms --------------------
    @torch.no_grad()
    def fit(self, y: TensorLike) -> LogStdNormalizer:
        """
        Fit mean and scale parameters from all observed probabilities (in log-domain).
        """
        pl_t = self._to_tensor(y, dtype=torch.float32).view(-1)
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
    def transform(self, y: TensorLike) -> torch.Tensor:
        """
        Transform from probability domain → standardized latent domain:
            z = (log(pl + eps) - mean) / scale
        """
        assert self._fitted, "Call fit(pl_train) before transform."
        pl_t = self._to_tensor(y, dtype=torch.float32)
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

    def set_eps(self, eps: float):
        """Update epsilon used in log(pl + eps)."""
        self.stats.eps = float(eps)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "stats": asdict(self.stats),
            "fitted": self._fitted,
            "min_prob": self.min_prob,
            "max_prob": self.max_prob,
            "device": self.device,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        s = state["stats"]
        self.stats = LogStdStats(
            mean=float(s["mean"]), scale=float(s["scale"]), eps=float(s["eps"])
        )
        self._fitted = bool(state.get("fitted", True))
        self.min_prob = float(state.get("min_prob", self.min_prob))
        self.max_prob = float(state.get("max_prob", self.max_prob))
        self.device = state.get("device", self.device)


class StdNormalizer(Normalizer):
    """
    A normalizer for distance (d) which standardizes values to
        z = (d - mean) / std_dev

    As error rate (pl) scales exponentially with d, LogStdNormalizer is used for pl
    and linear standardization is used for d.
    """

    def __init__(self, device):
        super().__init__(device)
        self.stats: StdStats = StdStats(mean=0.0, scale=1.0)

    @torch.no_grad()
    def fit(self, y: TensorLike) -> StdNormalizer:
        d_t = self._to_tensor(y, dtype=torch.float32).view(-1)
        mean = d_t.mean().item()
        std = d_t.std(unbiased=False).item()

        if std < 1e-12:
            std = 1.0

        self.stats = StdStats(mean=float(mean), scale=float(std))
        self._fitted = True
        return self

    @torch.no_grad()
    def transform(self, y: TensorLike) -> torch.Tensor:
        assert self._fitted, "Call fit(d_train) before transform."

        d_t = self._to_tensor(y, dtype=torch.float32)
        z = (d_t - self.stats.mean) / self.stats.scale
        return z

    @torch.no_grad()
    def inverse_transform(self, z: TensorLike) -> torch.Tensor:
        assert self._fitted, "Call fit(d_train) before inverse_transform."

        z_t = self._to_tensor(z, dtype=torch.float32)
        d = z_t * self.stats.scale + self.stats.mean
        return d

    @torch.no_grad()
    def inverse_mean_std(
        self, mu_z: TensorLike, std_z: TensorLike
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._fitted, "Call fit(d_train) before inverse_mean_std."

        mu_z_t = self._to_tensor(mu_z, dtype=torch.float32)
        std_z_t = self._to_tensor(std_z, dtype=torch.float32).abs()

        mu = mu_z_t * self.stats.scale + self.stats.mean
        std = std_z_t * abs(self.stats.scale)

        return mu, std

    def state_dict(self) -> Dict[str, Any]:
        return {
            "stats": asdict(self.stats),
            "fitted": self._fitted,
            "device": self.device,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        s = state["stats"]
        self.stats = StdStats(mean=float(s["mean"]), scale=float(s["scale"]))
        self._fitted = bool(state.get("fitted", True))
        self.device = state.get("device", self.device)

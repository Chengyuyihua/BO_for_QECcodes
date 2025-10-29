"""
This file contains the implementation of the Bayesian Optimization (BO) loop
specialized for QEC (quantum error-correction) code search.

Design highlights:
- EI-based acquisition (with its own internal normalizer) + hill-climbing suggestion.
- A GPTrainer is used for "hot training" (no structural change of the GP).
- Online diagnostics per iteration: for each newly evaluated point, we log
  predicted vs. true metrics in both probability space (pl) and latent (z) space.
"""

import time
import numpy as np
import torch
import tqdm
import gpytorch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BO_on_QEC:
    """
    BO driver for QEC code search/evaluation.

    Workflow (one iteration):
      1) Set EI's best objective value based on current data.
      2) Suggest next candidate points via the provided `suggest_next` function.
      3) Before evaluating them, compute GP posterior in latent (z) space, then
         map to probability (pl) space using the acquisition's normalizer.
         These are used for online diagnostics against later true values.
      4) Evaluate the black-box `objective_function` on each suggested point to
         obtain (F, pl_true). Here F is the objective to maximize, pl_true is the
         true logical error rate (probability space).
      5) Append new data to the dataset, refit the normalizer on pl, and perform
         a warm training step on the GP (labels in z space).
      6) Update the incumbent best objective and parameters; write progress logs.

    Attributes recorded:
      - self.pred_vs_true_history: a per-iteration list of dicts, each comparing
        predicted (mu/std) vs true values in both pl-space and z-space; also
        residuals and 68%/95% coverage indicators in pl-space.

    Notes:
      - `gp`: a gpytorch model (with or without `.likelihood`) supporting
        posterior prediction (via `posterior(...)` or forward call).
      - `gp_trainer`: an external trainer object that manages freezing schedules,
        rescaling when the normalizer changes, and performs a single training
        round on demand.
      - `acquisition_function`: provides (i) EI computation elsewhere,
        (ii) a normalizer with `.update_normalizer(pl)` and
        `.targets_from_pl(pl)` to map probabilities to latent z-space,
        (iii) `inverse_mean_std(mu_z, std_z)` to map z-posterior to pl-space,
        and (iv) `.set_best_value(best_F)`.
      - `suggest_next(gp)`: returns a `torch.float32 [n, d]` tensor on device.
      - `objective_function(x_np_single) -> (F, pl_total)`: black-box evaluation.

    Parameters
    ----------
    gp : gpytorch.models.ExactGP or compatible model
    gp_trainer : object
        Trainer object with methods: on_normalizer_change, update_data,
        maybe_freeze_unfreeze, train_one_round.
    acquisition_function : object
        EI-like acquisition with internal normalizer and helper transforms.
    suggest_next : Callable[[gp], torch.Tensor]
        Suggestion function returning candidate points (shape [n, d]).
    objective_function : Callable[[np.ndarray], Tuple[float, float]]
        Black-box evaluation that returns (objective F, probability pl).
    initial_X, initial_pl, initial_y : torch.Tensor
        Initial dataset (X in design space, pl in probability space, y=objective).
    BO_iterations : int
        Number of BO iterations to run.
    description : str
        A short description used in progress bars and logs.
    device : str
        Torch device string (default "cpu").
    pretrain : bool
        If True, fit normalizer with initial pl, sync trainer, and perform
        one warmup training round before BO starts.
    """
    def __init__(self,
                 gp,
                 gp_trainer,                 # GPTrainer instance
                 acquisition_function,       # EI with internal normalizer
                 suggest_next,
                 objective_function,         # (x_single_np) -> (F, pL_total)
                 initial_X: torch.Tensor,
                 initial_pl: torch.Tensor,   # probability space (pl)
                 initial_y: torch.Tensor,    # objective space (F)
                 BO_iterations=10,
                 description='',
                 device="cpu",
                 pretrain=True):

        self.gp = gp
        self.trainer = gp_trainer
        self.objective_function = objective_function
        self.acquisition_function = acquisition_function
        self.suggest_next = suggest_next
        self.device = device

        self.X  = initial_X.to(self.device, dtype=torch.float32)
        self.pl = initial_pl.to(self.device, dtype=torch.float32)   # probability space labels
        self.y  = initial_y.to(self.device, dtype=torch.float32)    # objective values F

        self.BO_iterations = int(BO_iterations)
        self.description = description

        self.best_value = self.y.max()
        self.best_parameters = self.X[self.y.argmax()].clone()
        print(f'Initial best value: {self.best_value.item():.4f}')

        # Online diagnostic log: per-iteration list of records (dicts)
        self.pred_vs_true_history = []

        # Optional pretraining on initial data (fits normalizer and runs one GP training round)
        if pretrain:
            # 1) Fit normalizer on initial pl
            self.acquisition_function.update_normalizer(self.pl)
            stats = self.acquisition_function.normalizer.get_stats()
            # 2) Prepare training labels in latent z space
            z_train = self.acquisition_function.targets_from_pl(self.pl)
            # 3) Notify trainer about normalizer rescale (amplitude adjustment)
            self.trainer.on_normalizer_change(old_std=None, new_std=stats.scale)
            # 4) Update data & optionally freeze/unfreeze model parts
            self.trainer.update_data(self.X, z_train)
            self.trainer.maybe_freeze_unfreeze(round_idx=0)
            # 5) One warmup training round
            _ = self.trainer.train_one_round()
            # 6) Set EI's incumbent best F
            self.acquisition_function.set_best_value(self.best_value)

    # ----------------- Utility: batch posterior in latent z-space -----------------
    @torch.no_grad()
    def _latent_posterior(self, X_tensor: torch.Tensor):
        """
        Compute GP posterior mean/std in latent (z) space for a batch of points.

        Returns
        -------
        mu : torch.Tensor, shape [n]
            Posterior mean in z space.
        std : torch.Tensor, shape [n]
            Posterior std in z space.
        """
        self.gp.eval()
        if hasattr(self.gp, "likelihood") and self.gp.likelihood is not None:
            self.gp.likelihood.eval()
        X_tensor = X_tensor.to(self.device, dtype=torch.float32)
        with gpytorch.settings.fast_pred_var():
            if hasattr(self.gp, "posterior"):
                post = self.gp.posterior(X_tensor)
                mu = post.mean.reshape(-1)
                var = post.variance.reshape(-1)
            else:
                mvn = self.gp(X_tensor)
                mu = mvn.mean.reshape(-1)
                var = mvn.variance.reshape(-1)
        std = var.clamp_min(1e-12).sqrt()
        return mu, std

    def run(self):
        """
        Execute the BO loop for `self.BO_iterations` iterations.

        Returns
        -------
        best_parameters : torch.Tensor
            The incumbent best design point.
        best_value : torch.Tensor (scalar)
            The best objective value found so far.
        evaluation_history : list[list[float]]
            Per-iteration list of objective values for the evaluated batch.
        """
        evaluation_history = []
        best_y_paras = []
        pbar = tqdm.tqdm(range(self.BO_iterations), desc=self.description)

        # TSV for online diagnostics (pred vs. true); easy to inspect externally.
        metrics_tsv = open('pred_vs_true.tsv', 'a', encoding='utf-8')
        if metrics_tsv.tell() == 0:
            metrics_tsv.write("round\tidx\tmu_pl\tstd_pl\tpl_true\tr_pl\tz_true\tmu_z\tstd_z\tr_z\tcover68_pl\tcover95_pl\n")

        for rnd in pbar:
            # --- Step 0: refresh EI's incumbent best objective ---
            self.acquisition_function.set_best_value(self.best_value)

            # --- Step 1: suggest candidates ---
            t0 = time.time()
            next_points = self.suggest_next(self.gp)  # torch.float32, [n, d], on device
            t_next = time.time()

            # --- Step 1b: pre-evaluation predictions for diagnostics ---
            # Use the current normalizer/GP params at the start of this round.
            mu_z_pred, std_z_pred = self._latent_posterior(next_points)  # [n]
            # Closed-form mapping from (mu_z, std_z) to (mu_pl, std_pl) under log-normal assumption.
            if not self.acquisition_function.normalizer.is_fitted():
                # Should not happen in practice (pretraining/previous rounds fit it).
                self.acquisition_function.update_normalizer(self.pl)
            mu_pl_pred, std_pl_pred = self.acquisition_function.normalizer.inverse_mean_std(mu_z_pred, std_z_pred)
            mu_pl_pred = mu_pl_pred.clamp(1e-12, 1.0 - 1e-12)
            std_pl_pred = std_pl_pred.clamp_min(1e-12)

            # --- Step 2: single-sample evaluations (objective & true pl) ---
            y_list, pl_list = [], []
            next_points_np = next_points.detach().round().clamp(0, 1).cpu().numpy().astype("int64")
            for i in range(next_points_np.shape[0]):
                y_i, pl_i = self.objective_function(next_points_np[i])
                y_list.append(float(y_i) if torch.is_tensor(y_i) else y_i)
                pl_list.append(float(pl_i) if torch.is_tensor(pl_i) else pl_i)

            next_values = torch.tensor(y_list,  dtype=torch.float32, device=self.device)  # objective F
            next_pl     = torch.tensor(pl_list, dtype=torch.float32, device=self.device)  # true pl
            t_evaluated = time.time()

            evaluation_history.append(next_values.detach().cpu().tolist())

            # --- Step 2b: record online diagnostics (pred vs. true) ---
            iter_metrics = []
            # Map true pl to z-space using the same normalizer.
            z_true_all = self.acquisition_function.normalizer.transform(next_pl).detach()
            for i in range(next_pl.numel()):
                mu_z = mu_z_pred[i]; sz = std_z_pred[i].clamp_min(1e-12)
                mu_pl = mu_pl_pred[i]; sp = std_pl_pred[i].clamp_min(1e-12)
                pl_t = next_pl[i]
                z_t = z_true_all[i]

                r_pl = float(((pl_t - mu_pl) / sp).item())  # standardized residual in pl-space
                r_z  = float(((z_t  - mu_z)  / sz).item())  # standardized residual in z-space
                cover68 = abs(r_pl) <= 1.0
                cover95 = abs(r_pl) <= 1.96

                m_rec = {
                    "round": int(rnd),
                    "idx": int(i),
                    "mu_pl": float(mu_pl.item()),
                    "std_pl": float(sp.item()),
                    "pl_true": float(pl_t.item()),
                    "r_pl": r_pl,
                    "z_true": float(z_t.item()),
                    "mu_z": float(mu_z.item()),
                    "std_z": float(sz.item()),
                    "r_z": r_z,
                    "cover68_pl": bool(cover68),
                    "cover95_pl": bool(cover95),
                }
                iter_metrics.append(m_rec)
                # Append a TSV line
                metrics_tsv.write(
                    f"{m_rec['round']}\t{m_rec['idx']}\t{m_rec['mu_pl']:.6e}\t{m_rec['std_pl']:.6e}\t"
                    f"{m_rec['pl_true']:.6e}\t{m_rec['r_pl']:.3f}\t{m_rec['z_true']:.6e}\t"
                    f"{m_rec['mu_z']:.6e}\t{m_rec['std_z']:.6e}\t{m_rec['r_z']:.3f}\t"
                    f"{int(m_rec['cover68_pl'])}\t{int(m_rec['cover95_pl'])}\n"
                )
            metrics_tsv.flush()
            self.pred_vs_true_history.append(iter_metrics)

            # --- Step 3: expand dataset with new points ---
            self.X  = torch.cat((self.X,  next_points), dim=0)
            self.y  = torch.cat((self.y,  next_values), dim=0)
            self.pl = torch.cat((self.pl, next_pl),     dim=0)

            # --- Step 3b: refit normalizer on latest pl and warm-train GP ---
            t1 = time.time()
            # Get old std before refit (for trainer's amplitude rescaling)
            try:
                old_std = self.acquisition_function.normalizer.get_stats().scale
            except Exception:
                old_std = None
            # Refit scaler with updated pl
            self.acquisition_function.update_normalizer(self.pl)
            new_std = self.acquisition_function.normalizer.get_stats().scale

            # Prepare z-space labels for training
            z_train = self.acquisition_function.targets_from_pl(self.pl)
            # Notify trainer about scaler change
            self.trainer.on_normalizer_change(old_std=old_std, new_std=new_std)
            # Update data, run freeze/unfreeze policy, and train one round
            self.trainer.update_data(self.X, z_train)
            self.trainer.maybe_freeze_unfreeze(round_idx=rnd + 1)
            _ = self.trainer.train_one_round()
            t_train = time.time()

            # --- Step 4: update incumbent best (objective space) ---
            nb = next_values.max()
            nb_idx = next_values.argmax().item()
            np_best = next_points[nb_idx]
            if nb.item() > self.best_value.item():
                self.best_value = nb
                self.best_parameters = np_best.clone()
                best_y_paras.append([
                    self.best_value.item(),
                    self.best_parameters.detach().cpu().numpy().tolist()
                ])
                print(f"new code updated: {[best_y_paras[-1][0], best_y_paras[-1][1]]}")

            # --- Step 5: progress bar & logging ---
            pbar.set_description(
                f'{self.description} (Best value: {self.best_value.item():.4f}), '
                f'time: suggest {t_next - t0:.3f}, eval {t_evaluated - t_next:.3f}, train {t_train - t1:.3f}'
            )
            with open('result.txt', 'a', encoding='utf-8') as f:
                f.write(
                    f"{self.description} (Best value: {self.best_value.item():.6f}), "
                    f"time: suggesting {t_next - t0:.6f}, evaluate {t_evaluated - t_next:.6f}, training {t_train - t1:.6f}\n"
                )

        metrics_tsv.close()
        print(f"Evaluation_history (len={len(evaluation_history)}).")
        # You can read detailed diagnostics from `bo.pred_vs_true_history` externally.
        return self.best_parameters, self.best_value, evaluation_history

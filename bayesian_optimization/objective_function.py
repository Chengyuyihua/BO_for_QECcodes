import math
import numpy as np
from evaluation.decoder_based_evaluation import *
from code_construction.code_construction import *
from evaluation.circuit_level_noise import MonteCarloEstimationOfLogicalErrorRateUnderCircuitLevelNoise

# =========================
# Utilities
# =========================

def _log_binom(n: int, j: int) -> float:
    """Stable log binomial: log C(n, j) via lgamma."""
    return math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)

def _build_pchip(x, y):
    """
    Try to build a PCHIP interpolator; fall back to linear if SciPy is missing.
    Returns (interp_obj, linear_fallback_flag).
    """
    try:
        from scipy.interpolate import PchipInterpolator
        return PchipInterpolator(np.asarray(x, float), np.asarray(y, float), extrapolate=True), False
    except Exception:
        return (np.asarray(x, float), np.asarray(y, float)), True

def _eval_interp(interp_obj, xq):
    """Evaluate either a PCHIP object or a linear interpolator (x, y tuple)."""
    if isinstance(interp_obj, tuple):
        x, y = interp_obj
        return np.interp(np.asarray(xq, float), x, y, left=y[0], right=y[-1])
    else:
        return interp_obj(np.asarray(xq, float))

def _eval_piecewise_slope(x, y, tq):
    """
    Compute piecewise-constant slope for linear interpolation fallback.
    x, y: 1D strictly increasing arrays.
    tq: scalar or array.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    tq = np.asarray(tq, float)

    idx = np.searchsorted(x, tq, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    dx = x[idx + 1] - x[idx]
    dy = y[idx + 1] - y[idx]
    slope = dy / dx

    # Boundary handling: use edge slopes for out-of-range queries
    slope = np.where(tq <= x[0], (y[1] - y[0]) / (x[1] - x[0]), slope)
    slope = np.where(tq >= x[-1], (y[-1] - y[-2]) / (x[-1] - x[-2]), slope)
    return slope

# =========================
# f2_converter: t -> f2(t)
# =========================

class f2_converter:
    r"""
    Compute and interpolate:
        f2(t) = (1/n) * log2( sum_{j=0}^{t} C(n,j) * 3^j )
    over integer t in [0..n], providing cubic (PCHIP) interpolation
    for real-valued t, and derivative f2'(t).
    """

    def __init__(self, n: int, use_pchip: bool = True):
        self.n = int(n)
        ln3 = math.log(3.0)
        log_terms = np.array([_log_binom(self.n, j) + j * ln3 for j in range(self.n + 1)], dtype=float)

        # prefix log-sum-exp for sum_{j=0..t}
        log_prefix = np.full(self.n + 1, -np.inf, dtype=float)
        acc = -np.inf
        for j in range(self.n + 1):
            if acc == -np.inf:
                acc = log_terms[j]
            else:
                m = max(acc, log_terms[j])
                acc = m + math.log(math.exp(acc - m) + math.exp(log_terms[j] - m))
            log_prefix[j] = acc

        # f2(t) = (1/n) * log2(sum ...) = log_prefix / (n * ln 2)
        self.t_grid = np.arange(self.n + 1, dtype=float)
        self.f2_table = log_prefix / (self.n * math.log(2.0))

        # Interpolator setup
        if use_pchip:
            self._t2f2, self._linear = _build_pchip(self.t_grid, self.f2_table)
            if not self._linear:
                self._t2f2_deriv = self._t2f2.derivative()
            else:
                self._t2f2_deriv = None
        else:
            self._t2f2, self._linear = (self.t_grid, self.f2_table), True
            self._t2f2_deriv = None

    def t_to_f2(self, t):
        """Evaluate f2(t) for scalar or array inputs."""
        return _eval_interp(self._t2f2, t)

    def df2_dt(self, t):
        """Evaluate derivative f2'(t)."""
        t = np.asarray(t, float)
        if not self._linear and self._t2f2_deriv is not None:
            return self._t2f2_deriv(t)
        else:
            x, y = self._t2f2
            return _eval_piecewise_slope(x, y, t)

# =========================================
# pl_t_converter: t <-> pL (with derivative)
# =========================================

class pl_t_converter:
    r"""
    Convert between the correctable-weight proxy t and the logical error rate p_L,
    using cubic (PCHIP) interpolation in log10-space for numerical stability.
    Also exposes d/dt [log10 p_L(t)] for uncertainty propagation.
    """

    def __init__(self, n: int, p_phys: float = 0.01, use_pchip: bool = True):
        if not (0.0 < p_phys < 0.5):
            raise ValueError("p_phys should be in (0, 0.5).")
        self.n = int(n)
        self.p = float(p_phys)

        # ----- Precompute log PMF terms -----
        log_p = math.log(self.p)
        log_q = math.log(1.0 - self.p)
        log_pmf = np.array([_log_binom(self.n, j) + j * log_p + (self.n - j) * log_q
                            for j in range(self.n + 1)], dtype=float)

        # Right-tail sums: log_tail[j] = log sum_{i=j..n} pmf(i)
        log_tail = np.full(self.n + 2, -np.inf, dtype=float)
        log_tail[self.n] = log_pmf[self.n]
        for j in range(self.n - 1, -1, -1):
            m = max(log_pmf[j], log_tail[j + 1])
            log_tail[j] = m + math.log(math.exp(log_pmf[j] - m) + math.exp(log_tail[j + 1] - m))

        # p_L(t) ≈ tail at j = floor(t)+1; build table over integer t
        self.t_grid = np.arange(self.n + 1, dtype=float)
        log_pl_table = np.array([log_tail[int(t) + 1] for t in self.t_grid], dtype=float)
        self.log10_pl_table = log_pl_table / math.log(10.0)

        # ----- Build interpolators -----
        if use_pchip:
            self._t2log10pl, self._t_linear = _build_pchip(self.t_grid, self.log10_pl_table)
            if not self._t_linear:
                self._t2log10pl_deriv = self._t2log10pl.derivative()
            else:
                self._t2log10pl_deriv = None
        else:
            self._t2log10pl, self._t_linear = (self.t_grid, self.log10_pl_table), True
            self._t2log10pl_deriv = None

        # Inverse mapping: log10 pL -> t
        x_inv = self.log10_pl_table[::-1]
        y_inv = self.t_grid[::-1]
        if use_pchip:
            self._log10pl2t, self._inv_linear = _build_pchip(x_inv, y_inv)
        else:
            self._log10pl2t, self._inv_linear = (x_inv, y_inv), True

    def t_to_pl(self, t):
        """Map t (float or array) → p_L(t) via cubic (or linear) interpolation in log10 space."""
        log10_pl = _eval_interp(self._t2log10pl, t)
        return np.power(10.0, log10_pl)

    def dlog10pl_dt(self, t):
        """Derivative of log10 p_L(t) with respect to t."""
        t = np.asarray(t, float)
        if not self._t_linear and self._t2log10pl_deriv is not None:
            return self._t2log10pl_deriv(t)
        else:
            x, y = self._t2log10pl
            return _eval_piecewise_slope(x, y, t)

    def pl_to_t(self, k, pl):
        """
        Map p_L ∈ (0,1) → t via inverse interpolation. 'k' is unused but kept for API consistency.
        """
        arr = np.asarray(pl, dtype=float)
        if np.any((arr <= 0.0) | (arr >= 1.0)):
            raise ValueError("pl must be in (0, 1).")
        log10_pl = np.log10(arr)
        t_hat = _eval_interp(self._log10pl2t, log10_pl)
        return t_hat if np.ndim(pl) else float(t_hat)

# =========================================
# ObjectiveFunction (with uncertainty)
# =========================================

import math
import numpy as np
import torch

class ObjectiveFunction:
    """
    Objective: F(x) = R + f2(t_hat) - 1, where
      - R = k/n,
      - t_hat = f1^{-1}(pL_hat) with pL_hat = 1 - (1 - pL)^{1/k} (per logical qubit),
      - f2(t) = (1/n) * log2(sum_{j=0}^t C(n,j) 3^j).

    Includes uncertainty propagation from p_L mean/std to F mean/std:
      - pl_to_obj_with_std(self, x, pl_total_mean, pl_total_std, return_aux=False)
        returns (F_mean, F_std), or with aux diagnostics if return_aux=True.
    """

    def __init__(self, code_constructor, lambda_ = 1,pp=0.01, decoder_param={'trail': 10000,'max_error':100},
                 circuit_level_noise=False, circuit_param=None):
        self.code_constructor = code_constructor
        self.n = int(code_constructor.n)
        self.pp = float(pp)
        self.decoder_param = dict(decoder_param)
        self.lambda_ = lambda_

        # Converters
        self.pl_t_converter = pl_t_converter(self.n, p_phys=self.pp, use_pchip=True)
        self.f2_converter = f2_converter(self.n, use_pchip=True)

        self._last_k = None
        self.circuit_level_noise = circuit_level_noise
        if circuit_level_noise:
            if circuit_param is None:
                circuit_param = {
                    'noise_model':'SD6',
                    'num_workers' : 24,
                    'rounds':12,
                    'custom_error_model':{},
                    'decoder':'bplsd'
                }
            self.circuit_param = circuit_param

    # ---------------------------
    # Utility: ensure x is np.int64 0/1 vector
    # ---------------------------
    @staticmethod
    def _to_np_bits(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().to("cpu")
            if x.dtype.is_floating_point:
                arr = np.rint(x.numpy()).astype(np.int64)
            else:
                arr = x.numpy().astype(np.int64)
        elif isinstance(x, np.ndarray):
            arr = x.astype(np.int64, copy=False)
        else:
            arr = np.asarray(x, dtype=np.int64)
        return arr

    def ler(self, css):
        """Compute the logical error rate, either via circuit-level or decoder-based simulation."""
        if self.circuit_level_noise:
            mc = MonteCarloEstimationOfLogicalErrorRateUnderCircuitLevelNoise(
                css, noise_model=self.circuit_param['noise_model'],
                p=self.pp, rounds=self.circuit_param['rounds'],
                custom_error_model=self.circuit_param['custom_error_model'],
                decoder=self.circuit_param['decoder']
            )
            pL = mc.run(shots=self.decoder_param['trail'],
                        max_error=self.decoder_param['max_error'],
                        num_workers=self.circuit_param['num_worker'])
        else:
            evaluator = CSS_Evaluator(css.hx, css.hz)
            pL = evaluator.Get_logical_error_rate_Monte_Carlo(
                physical_error_rate=self.pp,
                xyz_bias=[1, 1, 1],
                trail=self.decoder_param.get('trail', 10000)
            )
            pL = min(pL,1-(1e-20))
            pL = max(pL,1e-20)
            return float(pL)

    def nller(self, x):
        """Compute negative log-likelihood (-log pL)."""
        css = self.code_constructor.construct(self._to_np_bits(x))
        if css.k == 0:
            return 1.0
        pL = self.ler(css)
        return float(-np.log(pL)), float(pL)

    def nllerpq(self, x):
        """Compute -log(1 - (1 - pL)^{1/k}), used in per-logical-qubit metrics."""
        css = self.code_constructor.construct(self._to_np_bits(x))
        if css.k == 0:
            return 1.0
        pL = self.ler(css)
        return float(-np.log(1.0 - (1.0 - pL) ** (1.0 / css.k))), float(pL)

    def lerpq(self, x):
        """Compute per-logical-qubit logical error rate."""
        css = self.code_constructor.construct(self._to_np_bits(x))
        if css.k == 0:
            return 1.0
        pL = self.ler(css)
        pL_per_lq = 1.0 - (1.0 - pL) ** (1.0 / css.k)
        return max(float(pL_per_lq), 1e-30)

    def psuedo_t(self, css):
        """Compute the pseudo-t (effective correctable weight) and total pL."""
        if css.k == 0:
            return 0.0, 1.0
        pL = self.ler(css)
        # pL_per_lq = 1.0 - (1.0 - pL) ** (1.0 / css.k)
        # pL_per_lq = max(float(pL_per_lq), 1e-30)
        t_hat = self.pl_t_converter.pl_to_t(k=None, pl=pL)
        return float(t_hat), float(pL)

    def forward(self, x):
        """Compute the scalar objective F(x) and the corresponding total pL."""
        css = self.code_constructor.construct(self._to_np_bits(x))
        k = int(css.k)
        self._last_k = k
        if k == 0:
            return float(-1.0), float(1.0)

        R = k / float(self.n)
        t_hat, pL_total = self.psuedo_t(css)
        f2_val = float(self.f2_converter.t_to_f2(t_hat))
        F = self.lambda_ * R + f2_val - 1.0
        return float(F), float(pL_total)

    # ---------------------------
    # Uncertainty propagation: from (pL_mean, pL_std) → (F_mean, F_std)
    # ---------------------------
    def nlpl_with_std(self, x, pl_total_mean, pl_total_std):
        """Compute (-log pL, propagated std)."""
        return float(-np.log(pl_total_mean)),float( pl_total_std / pl_total_mean)
    
    def nllerpq_with_std(self, x, pl_total_mean, pl_total_std):
        """Compute mean/std of -log(1 - (1 - pL)^{1/k})."""
        css = self.code_constructor.construct(self._to_np_bits(x))
        mean = -np.log(1-(1-pl_total_mean)**(1/css.k))
        std = pl_total_std*(1-pl_total_mean)**(1/css.k-1)/(css.k * (1-(1-pl_total_mean)**(1/css.k)))
        return float(mean), float(std)

    def pl_to_obj_with_std(self, x, pl_total_mean, pl_total_std, return_aux: bool = False):
        """
        Given the mean/std of the total logical error rate (pL_total),
        compute the mean/std of the objective F(x).

        Parameters
        ----------
        x : code parameters (torch / np / list)
        pl_total_mean : float
            Mean of total logical error rate.
        pl_total_std : float
            Standard deviation of total logical error rate.
        return_aux : bool, default False
            If True, return detailed intermediate variables.

        Returns
        -------
        (F_mean, F_std) or (F_mean, F_std, aux)
        """
        css = self.code_constructor.construct(self._to_np_bits(x))
        k = int(css.k)
        if k == 0:
            ret2 = (float(-1.0), float(0.0))
            if return_aux:
                return ret2 + ({"note": "k=0; returning (-1,0)."},)
            return ret2

        R = k / float(self.n)
        eps = 1e-30
        m_total = float(pl_total_mean)
        s_total = max(float(pl_total_std), 0.0)

        # Clamp probabilities to (eps, 1 - eps)
        m_total = min(max(m_total, eps), 1.0 - 1e-12)

        # Convert total → per-logical-qubit
        m_per = 1.0 - (1.0 - m_total) ** (1.0 / k)
        gprime = (1.0 / k) * (1.0 - m_total) ** (1.0 / k - 1.0)
        s_per = abs(gprime) * s_total

        # per → t_hat → f2 → F_mean
        t_hat = float(self.pl_t_converter.pl_to_t(k=None, pl=m_per))
        f2_val = float(self.f2_converter.t_to_f2(t_hat))
        F_mean = R + f2_val - 1.0

        # Chain derivative: dF/d(pL_total)
        dlog10pl_dt = float(self.pl_t_converter.dlog10pl_dt(t_hat))
        dplper_dt = math.log(10.0) * m_per * dlog10pl_dt
        dt_dplper = 0.0 if abs(dplper_dt) < 1e-300 else 1.0 / dplper_dt
        df2_dt = float(self.f2_converter.df2_dt(t_hat))
        dF_dptotal = df2_dt * dt_dplper * gprime

        F_std = abs(dF_dptotal) * s_total

        if return_aux:
            aux = {
                "k": k,
                "R": R,
                "pL_total_mean": m_total,
                "pL_total_std": s_total,
                "pL_per_mean": m_per,
                "pL_per_std": s_per,
                "t_hat": t_hat,
                "f2": f2_val,
                "dlog10pl_dt": dlog10pl_dt,
                "df2_dt": df2_dt,
                "dF_dptotal": dF_dptotal,
            }
            return float(F_mean), float(F_std), aux
        else:
            return float(F_mean), float(F_std)

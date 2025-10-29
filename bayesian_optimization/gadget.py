from typing import Optional, Sequence, Tuple, List
import numpy as np

try:
    import torch
except Exception:
    torch = None

# optional SciPy backend
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class ChainComplexGlobalFeaturesEncoder:


    def __init__(self,
                 return_torch: bool = True,
                 dtype: Optional["torch.dtype"] = None,
                 device: Optional["torch.device"] = None,
                 max_degree_bin: int = 10,
                 quantiles: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
                 # SVD options
                 spectral_backend: str = "auto",   # "auto" | "scipy" | "power"
                 power_iters: int = 200,
                 power_tol: float = 1e-6,
                 power_seed: int = 0,
                 # two-step & NB options
                 eig_iters: int = 200,
                 eig_tol: float = 1e-6,
                 nb_iters: int = 200,
                 nb_tol: float = 1e-6,
                 nb_seed: int = 1234,
                 # Hodge options
                 hodge_k: int = 8,
                 hodge_dense_threshold: int = 2048):
        self.return_torch = return_torch and (torch is not None)
        self.dtype = dtype if dtype is not None else (torch.float32 if self.return_torch else np.float32)
        self.device = device
        self.max_degree_bin = int(max_degree_bin)
        self.quantiles = tuple(float(q) for q in quantiles)

        self.spectral_backend = spectral_backend
        self.power_iters = int(power_iters)
        self.power_tol = float(power_tol)
        self.power_seed = int(power_seed)

        self.eig_iters = int(eig_iters)
        self.eig_tol = float(eig_tol)

        self.nb_iters = int(nb_iters)
        self.nb_tol = float(nb_tol)
        self.nb_seed = int(nb_seed)

        self.hodge_k = int(hodge_k)
        self.hodge_dense_threshold = int(hodge_dense_threshold)

        self.last_slices = {}

    # ---------- degrees ----------
    @staticmethod
    def _np_deg_arrays(h: np.ndarray):
        Hnz = (h != 0)
        var_deg = Hnz.sum(axis=0).astype(np.int64, copy=False).ravel()
        chk_deg = Hnz.sum(axis=1).astype(np.int64, copy=False).ravel()
        return var_deg, chk_deg

    def _hist_node_perspective(self, deg: np.ndarray) -> np.ndarray:
        D = self.max_degree_bin
        if deg.size == 0:
            return np.zeros(D + 2, dtype=np.float32)
        bidx = np.minimum(np.maximum(deg, 0), D + 1)
        counts = np.bincount(bidx, minlength=D + 2).astype(np.float64)
        hist = counts / counts.sum() if counts.sum() > 0 else counts
        return hist.astype(np.float32, copy=False)

    def _quantiles(self, deg: np.ndarray) -> np.ndarray:
        if deg.size == 0:
            return np.zeros(len(self.quantiles), dtype=np.float32)
        try:
            qv = np.quantile(deg, self.quantiles, method="linear")
        except TypeError:
            qv = np.quantile(deg, self.quantiles, interpolation="linear")
        return qv.astype(np.float32, copy=False)

    # ---------- sigma2(H) ----------
    @staticmethod
    def _binarize_float(h: np.ndarray) -> np.ndarray:
        return (h != 0).astype(np.float64, copy=False)

    @staticmethod
    def _sigma2_power(H: np.ndarray, iters: int, tol: float, seed: int) -> float:
        m, n = H.shape
        if m == 0 or n == 0:
            return 0.0
        rng = np.random.default_rng(seed)
        def AtA(v): return H.T @ (H @ v)

        v = rng.standard_normal(n); v /= (np.linalg.norm(v) + 1e-12)
        lam_old = 0.0
        for _ in range(iters):
            w = AtA(v); nrm = np.linalg.norm(w)
            if nrm < 1e-14: return 0.0
            v = w / nrm; lam = float(v @ AtA(v))
            if abs(lam - lam_old) <= tol * max(1.0, abs(lam)): break
            lam_old = lam
        v1 = v.copy()
        v = rng.standard_normal(n); v -= (v @ v1) * v1
        v /= (np.linalg.norm(v) + 1e-12)
        lam_old = 0.0
        for _ in range(iters):
            w = AtA(v); w -= (w @ v1) * v1; nrm = np.linalg.norm(w)
            if nrm < 1e-14: return 0.0
            v = w / nrm; lam = float(v @ AtA(v))
            if abs(lam - lam_old) <= tol * max(1.0, abs(lam)): break
            lam_old = lam
        return float(np.sqrt(max(lam, 0.0)))

    @staticmethod
    def _sigma2_scipy(H: np.ndarray) -> float:
        m, n = H.shape
        if m == 0 or n == 0:
            return 0.0
        Hs = sp.csr_matrix(H)
        try:
            s = spla.svds(Hs, k=2, which="LM", return_singular_vectors=False)
            s = np.sort(np.array(s, dtype=np.float64))
            return float(max(s[-2] if s.size >= 2 else 0.0, 0.0))
        except Exception:
            return ChainComplexGlobalFeaturesEncoder._sigma2_power(H, iters=200, tol=1e-6, seed=0)

    def _sigma2_pair(self, Hx: np.ndarray, Hz: np.ndarray) -> Tuple[float, float]:
        Hx_bin = self._binarize_float(Hx)
        Hz_bin = self._binarize_float(Hz)
        use_scipy = (_HAS_SCIPY and self.spectral_backend in ("auto", "scipy"))
        if use_scipy:
            return self._sigma2_scipy(Hx_bin), self._sigma2_scipy(Hz_bin)
        return (self._sigma2_power(Hx_bin, self.power_iters, self.power_tol, self.power_seed),
                self._sigma2_power(Hz_bin, self.power_iters, self.power_tol, self.power_seed + 1))

    # ---------- lambda2 on variable–variable 2-step ----------
    @staticmethod
    def _build_varvar_norm_adj_sparse(H: np.ndarray):
        Hs = sp.csr_matrix((H != 0).astype(np.int8))
        A = Hs.T @ Hs
        A.setdiag(0); A.eliminate_zeros()
        if A.shape[0] == 0:
            return sp.csr_matrix((0, 0)), np.array([], dtype=bool)
        deg = np.array(A.sum(axis=1)).ravel()
        mask = deg > 0
        if mask.sum() == 0:
            return sp.csr_matrix((0, 0)), mask
        A = A[mask][:, mask]; deg = deg[mask]
        dinv = 1.0 / np.sqrt(deg)
        A = A.tocoo()
        data = A.data * dinv[A.row] * dinv[A.col]
        An = sp.csr_matrix((data, (A.row, A.col)), shape=A.shape)
        return An, mask

    @staticmethod
    def _eig2_symmetric_power_dense(An: np.ndarray, iters: int, tol: float, seed: int) -> float:
        n = An.shape[0]
        if n == 0: return 0.0
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(n); v /= (np.linalg.norm(v) + 1e-12)
        lam_old = 0.0
        for _ in range(iters):
            w = An @ v; nrm = np.linalg.norm(w)
            if nrm < 1e-14: return 0.0
            v = w / nrm; lam = float(v @ (An @ v))
            if abs(lam - lam_old) <= tol * max(1.0, abs(lam)): break
            lam_old = lam
        v1 = v.copy()
        v = rng.standard_normal(n); v -= (v @ v1) * v1
        v /= (np.linalg.norm(v) + 1e-12)
        lam_old = 0.0
        for _ in range(iters):
            w = An @ v; w -= (w @ v1) * v1; nrm = np.linalg.norm(w)
            if nrm < 1e-14: return 0.0
            v = w / nrm; lam = float(v @ (An @ v))
            if abs(lam - lam_old) <= tol * max(1.0, abs(lam)): break
            lam_old = lam
        return float(min(1.0, max(-1.0, lam)))

    def _lambda2_varvar(self, H: np.ndarray) -> float:
        Hbin = (H != 0).astype(np.float64, copy=False)
        if _HAS_SCIPY:
            An, _ = self._build_varvar_norm_adj_sparse(Hbin)
            if An.shape[0] == 0:
                return 0.0
            try:
                vals = spla.eigsh(An, k=2, which="LM", return_eigenvectors=False)
                vals = np.sort(np.abs(vals))
                return float(min(1.0, max(0.0, vals[-2]))) if vals.size >= 2 else float(vals[-1])
            except Exception:
                Ad = An.toarray()
                return self._eig2_symmetric_power_dense(Ad, self.eig_iters, self.eig_tol, seed=42)
        else:
            A = Hbin.T @ Hbin
            np.fill_diagonal(A, 0.0)
            deg = A.sum(axis=1)
            mask = deg > 0
            if not mask.any(): return 0.0
            A = A[np.ix_(mask, mask)]; deg = deg[mask]
            dinv = 1.0 / np.sqrt(deg)
            An = (A * dinv).T * dinv
            return self._eig2_symmetric_power_dense(An, self.eig_iters, self.eig_tol, seed=42)

    def _lambda2_varvar_pair(self, Hx: np.ndarray, Hz: np.ndarray) -> Tuple[float, float]:
        return self._lambda2_varvar(Hx), self._lambda2_varvar(Hz)

    # ---------- non-backtracking spectral radius ----------
    @staticmethod
    def _nb_build_edge_lists(H: np.ndarray) -> Tuple[List[List[int]], List[List[int]], np.ndarray, np.ndarray]:
        Hnz = (H != 0)
        m, n = Hnz.shape
        N_v = [list(np.where(Hnz[:, j])[0]) for j in range(n)]
        N_c = [list(np.where(Hnz[i, :])[0]) for i in range(m)]
        var_idx = []
        chk_idx = []
        for j in range(n):
            for i in N_v[j]:
                var_idx.append(j); chk_idx.append(i)
        return N_v, N_c, np.asarray(var_idx, dtype=np.int64), np.asarray(chk_idx, dtype=np.int64)

    def _nonbacktracking_rho(self, H: np.ndarray) -> float:
        N_v, N_c, var_idx, chk_idx = self._nb_build_edge_lists(H)
        E = var_idx.size
        if E == 0: return 0.0
        rng = np.random.default_rng(self.nb_seed)
        a = rng.standard_normal(E); b = rng.standard_normal(E)
        edges_of_var = [[] for _ in range(len(N_v))]
        edges_of_chk = [[] for _ in range(len(N_c))]
        for e in range(E):
            v = var_idx[e]; c = chk_idx[e]
            edges_of_var[v].append(e); edges_of_chk[c].append(e)

        def step(a, b):
            sum_a_c = np.zeros(len(N_c))
            for c, edges in enumerate(edges_of_chk):
                if edges: sum_a_c[c] = a[edges].sum()
            new_b = sum_a_c[chk_idx] - a
            sum_b_v = np.zeros(len(N_v))
            for v, edges in enumerate(edges_of_var):
                if edges: sum_b_v[v] = new_b[edges].sum()
            new_a = sum_b_v[var_idx] - new_b
            return new_a, new_b

        prev_norm = np.linalg.norm(np.concatenate([a, b]))
        if prev_norm < 1e-12:
            a[:] = 1.0; b[:] = 1.0; prev_norm = np.linalg.norm(np.concatenate([a, b]))

        rho = 0.0
        for _ in range(self.nb_iters):
            new_a, new_b = step(a, b)
            y = np.concatenate([new_a, new_b])
            y_norm = np.linalg.norm(y)
            if y_norm < 1e-18: return 0.0
            rho_new = y_norm / prev_norm
            a, b = new_a / y_norm, new_b / y_norm
            if abs(rho_new - rho) <= self.nb_tol * max(1.0, abs(rho_new)):
                rho = rho_new; break
            rho = rho_new; prev_norm = 1.0
        return float(rho)

    def _nonbacktracking_rho_pair(self, Hx: np.ndarray, Hz: np.ndarray) -> Tuple[float, float]:
        return self._nonbacktracking_rho(Hx), self._nonbacktracking_rho(Hz)

    # ---------- 4-cycle counts ----------
    @staticmethod
    def _count_c4(H: np.ndarray) -> int:
        """
        Count 4-cycles in the bipartite Tanner graph of H.
        Uses sum_{i<j} C( (H^T H)_{ij}, 2 ).
        """
        Hbin = (H != 0).astype(np.int64, copy=False)
        if _HAS_SCIPY:
            Hs = sp.csr_matrix(Hbin)
            B = (Hs.T @ Hs).tocsr()
            B.setdiag(0); B.eliminate_zeros()
            # use only upper triangle
            Bu = sp.triu(B, k=1).tocoo()
            v = Bu.data.astype(np.int64, copy=False)
            return int(np.sum(v * (v - 1) // 2))
        else:
            B = Hbin.T @ Hbin
            np.fill_diagonal(B, 0)
            iu = np.triu_indices(B.shape[0], k=1)
            v = B[iu]
            return int(np.sum(v * (v - 1) // 2))

    @staticmethod
    def _count_c4_mixed(Hx: np.ndarray, Hz: np.ndarray) -> int:
        """
        Mixed 4-cycles: variable pair shares one X-check and one Z-check.
        Count = sum_{i<j} ( (Hx^T Hx)_{ij} * (Hz^T Hz)_{ij} ).
        """
        X = (Hx != 0).astype(np.int64, copy=False)
        Z = (Hz != 0).astype(np.int64, copy=False)
        if _HAS_SCIPY:
            Xs = sp.csr_matrix(X); Zs = sp.csr_matrix(Z)
            BX = (Xs.T @ Xs).tocsr(); BZ = (Zs.T @ Zs).tocsr()
            for M in (BX, BZ):
                M.setdiag(0); M.eliminate_zeros()
            # upper triangle product
            BXu = sp.triu(BX, k=1).tocoo()
            # Pull corresponding entries from BZ (sparse gather)
            vals = BZ.tocsr()[BXu.row, BXu.col].A.ravel().astype(np.int64, copy=False)
            return int(np.sum(BXu.data.astype(np.int64) * vals))
        else:
            BX = X.T @ X; BZ = Z.T @ Z
            np.fill_diagonal(BX, 0); np.fill_diagonal(BZ, 0)
            iu = np.triu_indices(BX.shape[0], k=1)
            return int(np.sum(BX[iu] * BZ[iu]))

    def _count_c4_all(self, Hx: np.ndarray, Hz: np.ndarray) -> Tuple[int, int, int]:
        return self._count_c4(Hx), self._count_c4(Hz), self._count_c4_mixed(Hx, Hz)

    # ---------- Hodge / L1 smallest eigenvalues ----------
    def _hodge_L1_small_eigs(self, Hx: np.ndarray, Hz: np.ndarray) -> np.ndarray:
        """
        Compute the smallest hodge_k eigenvalues of L1 = Hx^T Hx + Hz^T Hz (float).
        Uses sparse eigsh when available; falls back to dense eigh for small n.
        """
        HxF = (Hx != 0).astype(np.float64, copy=False)
        HzF = (Hz != 0).astype(np.float64, copy=False)
        n = HxF.shape[1]
        if n == 0 or self.hodge_k <= 0:
            return np.zeros(0, dtype=np.float32)

        k = min(self.hodge_k, max(1, n - 1))  # eigsh requires k < n
        if _HAS_SCIPY:
            Xs = sp.csr_matrix(HxF); Zs = sp.csr_matrix(HzF)
            L1 = (Xs.T @ Xs) + (Zs.T @ Zs)   # n x n sparse, PSD
            try:
                vals = spla.eigsh(L1, k=k, which="SA", return_eigenvectors=False)
                vals = np.sort(np.maximum(vals, 0.0))
            except Exception:
                # fallback dense
                L1d = L1.toarray()
                vals = np.linalg.eigvalsh(L1d)
                vals = np.sort(np.maximum(vals, 0.0))[:k]
        else:
            # dense path; guard big n
            L1 = HxF.T @ HxF + HzF.T @ HzF
            if n <= self.hodge_dense_threshold:
                vals = np.linalg.eigvalsh(L1)
                vals = np.sort(np.maximum(vals, 0.0))[:k]
            else:
                # crude randomized subspace iteration for smallest (shift by epsilon)
                eps = 1e-6
                M = np.linalg.pinv(L1 + eps * np.eye(n))  # inverse approximates smallest of L1
                # now largest eigenvalues of M ≈ 1/(smallest of L1)
                r = min(k, 8)
                V = np.random.default_rng(0).standard_normal((n, r))
                for _ in range(20):
                    V, _ = np.linalg.qr(M @ V)
                S = V.T @ (L1 @ V)
                vals = np.linalg.eigvalsh(S)
                vals = np.sort(np.maximum(vals, 0.0))[:k]

        # pad / cast
        out = np.zeros(self.hodge_k, dtype=np.float32)
        out[:len(vals)] = vals.astype(np.float32)
        return out

    # ---------- main encode ----------
    def encode(self, c) -> "np.ndarray | torch.Tensor":
        feats = []
        slices = {}

        # Base
        n = int(c.n); k = int(c.k)
        rank_hx = int(c.gf2_rank(c.hx))
        rank_hz = int(c.gf2_rank(c.hz))
        base = np.array([n, k, rank_hx, rank_hz], dtype=np.float32)
        slices["base"] = (len(feats), len(feats) + base.size)
        feats.append(base)

        # Degrees
        vx, cx = self._np_deg_arrays(c.hx)
        vz, cz = self._np_deg_arrays(c.hz)
        vmerge = (vx + vz).astype(np.int64, copy=False)

        for key, arr in [
            ("vX_hist", self._hist_node_perspective(vx)),
            ("cX_hist", self._hist_node_perspective(cx)),
            ("vZ_hist", self._hist_node_perspective(vz)),
            ("cZ_hist", self._hist_node_perspective(cz)),
            ("vM_hist", self._hist_node_perspective(vmerge)),
        ]:
            slices[key] = (len(feats), len(feats) + arr.size); feats.append(arr)

        for key, arr in [
            ("vX_q", self._quantiles(vx)), ("cX_q", self._quantiles(cx)),
            ("vZ_q", self._quantiles(vz)), ("cZ_q", self._quantiles(cz)),
            ("vM_q", self._quantiles(vmerge)),
        ]:
            slices[key] = (len(feats), len(feats) + arr.size); feats.append(arr)

        # Spectral: sigma2
        s2x, s2z = self._sigma2_pair(c.hx, c.hz)
        spec_sigma2 = np.array([s2x, s2z], dtype=np.float32)
        slices["spec_sigma2"] = (len(feats), len(feats) + spec_sigma2.size); feats.append(spec_sigma2)

        # Spectral: lambda2 (var–var)
        l2x, l2z = self._lambda2_varvar_pair(c.hx, c.hz)
        spec_l2vv = np.array([l2x, l2z], dtype=np.float32)
        slices["spec_lambda2_vv"] = (len(feats), len(feats) + spec_l2vv.size); feats.append(spec_l2vv)

        # Spectral: non-backtracking rho
        rho_x, rho_z = self._nonbacktracking_rho_pair(c.hx, c.hz)
        spec_nb = np.array([rho_x, rho_z], dtype=np.float32)
        slices["spec_nb_rho"] = (len(feats), len(feats) + spec_nb.size); feats.append(spec_nb)

        # 4-cycles: [X, Z, mixed]
        c4x, c4z, c4m = self._count_c4_all(c.hx, c.hz)
        c4 = np.array([c4x, c4z, c4m], dtype=np.float32)
        slices["cycles_4"] = (len(feats), len(feats) + c4.size); feats.append(c4)

        # Hodge / L1 smallest eigenvalues
        l1_small = self._hodge_L1_small_eigs(c.hx, c.hz)
        slices["hodge_L1_small"] = (len(feats), len(feats) + l1_small.size); feats.append(l1_small)

        vec = np.concatenate(feats, dtype=np.float32)
        self.last_slices = slices

        if self.return_torch:
            return torch.tensor(vec, dtype=self.dtype, device=self.device)
        return vec

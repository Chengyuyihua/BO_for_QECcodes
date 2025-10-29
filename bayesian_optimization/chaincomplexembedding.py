from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn




def _cum_offsets(sizes: List[int]) -> List[int]:
    """Compute cumulative offsets for a list of segment sizes."""
    offs = [0]
    s = 0
    for v in sizes:
        s += int(v)
        offs.append(s)
    return offs


class ChainComplexEmbedder(nn.Module):
    """
    Multi-view, multi-relation embedder built from stacks of ChainComplexMessagePassingLayer.

    This class accepts either:
      • a list of per-sample 'views' (each item is RelationEncoder.encode(css_code) output), or
      • a packed dict of views (block-diagonal batched graphs),
    and returns a (B, d_model) embedding for the batch.

    Args
    ----
    views_info : List[dict]
        Must match RelationEncoder's views_info:
          {"name": str, "partite_classes": List[str], "relations": List[str], ...}
    d_model : int
        Hidden size for node features (shared across node types).
    num_layers : int
        Number of message passing layers.
    view_aggr : {'sum','mean'}
        How to aggregate outputs from different views at the same depth.
    readout_types : Optional[List[str]]
        Node types to include in readout pooling. Defaults to all types across views_info.
    num_bases : Optional[int]
        Basis size for relation weights (None => per-relation weights).
    norm : {'sym','dst','none'}
        Per-relation normalization used inside the message passing layers.
    residual : bool
        Whether each message passing layer uses residual.
    self_loop : bool
        Whether each layer injects an additional gated self-loop message (besides residual).
    dropout : float
        Dropout rate used in layers and readout MLP.
    act : Optional[nn.Module]
        Activation used in the 2-layer MLP inside each message passing layer.
    """

    def __init__(self,
                 views_info: List[Dict[str, Any]],
                 d_model: int = 128,
                 num_layers: int = 4,
                 *,
                 view_aggr: str = "sum",
                 readout_types: Optional[List[str]] = None,
                 # per-layer options
                 num_bases: Optional[int] = 4,
                 norm: str = "sym",
                 residual: bool = True,
                 self_loop: bool = False,
                 dropout: float = 0.1,
                 act: Optional[nn.Module] = None):
        super().__init__()
        assert view_aggr in ("sum", "mean")
        self.view_aggr = view_aggr
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)

        # ---- Parse schema from views_info ----
        self.views_info = []
        self.view_names: List[str] = []
        self.view_node_types: Dict[str, List[str]] = {}
        self.view_relations: Dict[str, List[str]] = {}
        all_types = set()

        for v in views_info:
            name = v.get("name", "view")
            nts = list(v.get("partite_classes", []))
            rels = list(v.get("relations", []))
            self.views_info.append({"name": name, "partite_classes": nts, "relations": rels})
            self.view_names.append(name)
            self.view_node_types[name] = nts
            self.view_relations[name] = rels
            all_types.update(nts)

        self.all_node_types: List[str] = sorted(all_types)
        self.readout_types: List[str] = list(readout_types) if readout_types is not None else self.all_node_types

        # ---- Learnable initial token per node type ----
        self.type_tokens = nn.ParameterDict({
            t: nn.Parameter(torch.empty(1, self.d_model)) for t in self.all_node_types
        })
        for p in self.type_tokens.values():
            nn.init.xavier_uniform_(p)

        # ---- Cross-view gates: one scalar per (layer, view) ----
        self.view_gates = nn.ParameterList([
            nn.ParameterDict({vn: nn.Parameter(torch.tensor(1.0)) for vn in self.view_names})
            for _ in range(self.num_layers)
        ])

        # ---- Build message passing stacks (per layer, per view) ----
        Layer = ChainComplexMessagePassingLayer
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            per_view = nn.ModuleDict()
            for vn in self.view_names:
                per_view[vn] = Layer(
                    node_types=self.all_node_types,          # pass full set; layer skips empty types by 'sizes'
                    relations=self.view_relations[vn],
                    in_dim=self.d_model,
                    out_dim=self.d_model,
                    num_bases=num_bases,
                    aggr="sum",            # sum over relations; cross-view aggregation happens here
                    norm=norm,
                    residual=residual,
                    self_loop=self_loop,
                    dropout=dropout,
                    act=act if act is not None else nn.GELU(),
                )
            self.layers.append(per_view)

        # ---- Readout: pool selected types, concat, MLP ----
        read_dim = len(self.readout_types) * self.d_model
        hidden = max(read_dim, 128)
        self.readout_mlp = nn.Sequential(
            nn.LayerNorm(read_dim),
            nn.Linear(read_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.d_model),   # change if you want a different final embedding size
        )

    # ===================== Packing utilities =====================

    @staticmethod
    def _parse_relation_tag(tag: str) -> Tuple[str, str]:
        """Return (SRC, DST) from 'A_B' (accepts 'bA_B' / 'coA_B' and strips the prefix)."""
        t = tag.strip()
        if t.startswith("co"):
            t = t[2:]
        elif t.startswith("b"):
            t = t[1:]
        if "_" not in t:
            raise ValueError(f"Bad relation tag '{tag}'; expected 'A_B' (optionally prefixed by 'b'/'co').")
        a, b = t.split("_", 1)
        return a.upper(), b.upper()

    @staticmethod
    def _concat_sparse_blockdiag(
        parts: List[Optional[torch.Tensor]],
        dst_sizes: List[int],
        src_sizes: List[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build a block-diagonal sparse COO by concatenating sample-wise sparse matrices
        with row/col offsets. Each element in `parts` is a (|dst_i| x |src_i|) sparse COO or None.
        """
        assert len(parts) == len(dst_sizes) == len(src_sizes)
        B = len(parts)

        idx_rows: List[torch.Tensor] = []
        idx_cols: List[torch.Tensor] = []
        vals: List[torch.Tensor] = []

        roff = 0
        coff = 0
        for i in range(B):
            A_i = parts[i]
            if (A_i is not None) and (A_i._nnz() > 0):
                Ai = A_i.coalesce().cpu()  # pack on CPU; move to device at the end
                idx = Ai.indices()
                val = Ai.values().to(dtype=torch.float32)
                idx_rows.append(idx[0] + roff)
                idx_cols.append(idx[1] + coff)
                vals.append(val)
            roff += int(dst_sizes[i])
            coff += int(src_sizes[i])

        total_dst = sum(int(x) for x in dst_sizes)
        total_src = sum(int(x) for x in src_sizes)

        if len(vals) == 0:
            # empty sparse tensor with the correct global shape
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_val = torch.empty((0,), dtype=dtype, device=device)
            return torch.sparse_coo_tensor(empty_idx, empty_val, size=(total_dst, total_src),
                                           device=device, dtype=dtype).coalesce()

        rows = torch.cat(idx_rows, dim=0)
        cols = torch.cat(idx_cols, dim=0)
        vcat = torch.cat(vals, dim=0)

        A = torch.sparse_coo_tensor(
            torch.stack([rows, cols], dim=0),
            vcat,
            size=(total_dst, total_src),
            device=device,
            dtype=dtype,
        ).coalesce()
        return A

    def _pack_batch(
        self,
        sample_views_list: List[Dict[str, Dict[str, Any]]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[int]]]:
        """
        Pack a list of per-sample 'views' into block-diagonal batched graphs for each view.

        Returns
        -------
        packed_views : Dict[view_name -> view_dict]
            Each view_dict contains:
              - 'sizes': {type: N_total}
              - 'adj'  : {relation_tag: sparse_coo (|DST_total| x |SRC_total|)}
              - 'deg'  : {relation_tag: (deg_src, deg_dst)} with concatenated vectors
        batch_splits : Dict[type -> List[int]]
            Per-type node counts per sample (used for per-sample pooling).
        """
        B = len(sample_views_list)

        # (1) Build per-type splits (max across views for robustness)
        batch_splits: Dict[str, List[int]] = {t: [0] * B for t in self.all_node_types}
        for b, views in enumerate(sample_views_list):
            for vn in self.view_names:
                vdict = views.get(vn, None)
                if vdict is None:
                    continue
                sz = vdict.get("sizes", {})
                for t in self.all_node_types:
                    batch_splits[t][b] = max(batch_splits[t][b], int(sz.get(t, 0)))

        # (2) Pack each view independently
        packed_views: Dict[str, Dict[str, Any]] = {}
        for vn in self.view_names:
            # total sizes per type in this view (sum over samples)
            totals_v: Dict[str, int] = {t: 0 for t in self.all_node_types}
            for b in range(B):
                vdict = sample_views_list[b].get(vn, None)
                sz = vdict.get("sizes", {}) if vdict is not None else {}
                for t in self.all_node_types:
                    totals_v[t] += int(sz.get(t, 0))

            adj_packed: Dict[str, torch.Tensor] = {}
            deg_packed: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

            for r in self.view_relations[vn]:
                src_t, dst_t = self._parse_relation_tag(r)

                # collect per-sample blocks and sizes
                parts: List[Optional[torch.Tensor]] = []
                dst_sizes: List[int] = []
                src_sizes: List[int] = []
                deg_src_list: List[torch.Tensor] = []
                deg_dst_list: List[torch.Tensor] = []

                for b in range(B):
                    vdict = sample_views_list[b].get(vn, None)
                    if vdict is None:
                        parts.append(None)
                        dst_sizes.append(batch_splits[dst_t][b])
                        src_sizes.append(batch_splits[src_t][b])
                        # zero degrees if the view is absent
                        if batch_splits[src_t][b] > 0:
                            deg_src_list.append(torch.zeros(batch_splits[src_t][b], dtype=torch.float32))
                        if batch_splits[dst_t][b] > 0:
                            deg_dst_list.append(torch.zeros(batch_splits[dst_t][b], dtype=torch.float32))
                        continue

                    A = vdict.get("adj", {}).get(r, None)
                    ds = int(vdict.get("sizes", {}).get(dst_t, 0))
                    ss = int(vdict.get("sizes", {}).get(src_t, 0))
                    parts.append(A if A is not None else None)
                    dst_sizes.append(ds)
                    src_sizes.append(ss)

                    # degrees: use provided if present, else zeros
                    deg_entry = vdict.get("deg", {}).get(r, None)
                    if deg_entry is None:
                        if ss > 0:
                            deg_src_list.append(torch.zeros(ss, dtype=torch.float32))
                        if ds > 0:
                            deg_dst_list.append(torch.zeros(ds, dtype=torch.float32))
                    else:
                        dsrc, ddst = deg_entry
                        deg_src_list.append(dsrc.detach().cpu())
                        deg_dst_list.append(ddst.detach().cpu())

                # block-diagonal adjacency
                A_batch = self._concat_sparse_blockdiag(parts, dst_sizes, src_sizes, device=device, dtype=dtype)

                # concatenated degrees
                deg_src_cat = torch.cat(deg_src_list, dim=0) if deg_src_list else torch.zeros(0, dtype=torch.float32)
                deg_dst_cat = torch.cat(deg_dst_list, dim=0) if deg_dst_list else torch.zeros(0, dtype=torch.float32)
                deg_src_cat = deg_src_cat.to(device)
                deg_dst_cat = deg_dst_cat.to(device)

                adj_packed[r] = A_batch
                deg_packed[r] = (deg_src_cat, deg_dst_cat)

            packed_views[vn] = {
                "sizes": totals_v,
                "adj": adj_packed,
                "deg": deg_packed,
            }

        return packed_views, batch_splits

    # ===================== Pooling & init utilities =====================

    @staticmethod
    def _gather_global_sizes(views: Dict[str, Dict[str, Any]],
                             all_types: List[str]) -> Dict[str, int]:
        """Collect total node counts per type across views (they should match; take max for robustness)."""
        totals: Dict[str, int] = {t: 0 for t in all_types}
        for v in views.values():
            sz = v.get("sizes", {})
            for t in all_types:
                totals[t] = max(totals[t], int(sz.get(t, 0)))
        return totals

    @staticmethod
    def _build_initial_H(totals: Dict[str, int],
                         type_tokens: nn.ParameterDict,
                         device: torch.device) -> Dict[str, torch.Tensor]:
        """Create initial features by repeating per-type tokens according to total node counts."""
        H: Dict[str, torch.Tensor] = {}
        for t, n in totals.items():
            if n <= 0:
                H[t] = torch.zeros((0, type_tokens[t].shape[1]), device=device)
            else:
                H[t] = type_tokens[t].to(device).expand(n, -1).clone()
        return H

    @staticmethod
    def _pool_type_by_splits(H_t: torch.Tensor, splits: Optional[List[int]], mode: str = "mean") -> torch.Tensor:
        """
        Per-type, per-sample pooling for a batched graph.

        H_t : (N_t, d)
        splits : [n1, n2, ..., nB] or None (None -> B=1)
        Returns (B, d)
        """
        if splits is None:
            # Single-sample case
            if H_t.numel() == 0:
                return H_t.new_zeros((1, H_t.shape[1]))
            if mode == "sum":
                return H_t.sum(dim=0, keepdim=True)
            return H_t.mean(dim=0, keepdim=True)

        B = len(splits)
        offs = _cum_offsets(splits)
        out = H_t.new_zeros((B, H_t.shape[1]))
        for b in range(B):
            s, e = offs[b], offs[b + 1]
            if e > s:
                if mode == "sum":
                    out[b] = H_t[s:e].sum(dim=0)
                else:
                    out[b] = H_t[s:e].mean(dim=0)
            else:
                out[b] = 0.0
        return out

    # ===================== Forward =====================

    def forward(self,
                views_or_list: Any,
                batch_splits: Optional[Dict[str, List[int]]] = None) -> torch.Tensor:
        """
        Accept either:
          • a list of per-sample views (output of RelationEncoder.encode for each sample), or
          • a packed dict of views (block-diagonal batched graphs).
        Returns:
          Tensor of shape (B, d_model).
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # If a list is provided, pack it first
        if isinstance(views_or_list, list):
            packed_views, batch_splits = self._pack_batch(views_or_list, device=device, dtype=dtype)
            views = packed_views
        else:
            views = views_or_list
            # If no batch_splits are provided, assume a single-sample batch (B=1)
            if batch_splits is None:
                totals_single = self._gather_global_sizes(views, self.all_node_types)
                batch_splits = {t: [int(totals_single.get(t, 0))] for t in self.all_node_types}

        # (1) Collect total node counts per type and create initial features
        totals = self._gather_global_sizes(views, self.all_node_types)
        H = self._build_initial_H(totals, self.type_tokens, device=device)

        # (2) L stacked layers; per layer: run every view, then aggregate across views
        for li, per_view in enumerate(self.layers):
            outs_per_view: Dict[str, Dict[str, torch.Tensor]] = {}
            for vn in self.view_names:
                if vn not in views:
                    continue
                outs_per_view[vn] = per_view[vn](H, views[vn])

            # Cross-view aggregation into the next H
            H_new: Dict[str, torch.Tensor] = {}
            for t in self.all_node_types:
                acc = None
                count = 0
                for vn, out_v in outs_per_view.items():
                    if t in out_v:
                        w = self.view_gates[li][vn]
                        contrib = out_v[t] * w
                        acc = contrib if acc is None else (acc + contrib)
                        count += 1
                if acc is None:
                    H_new[t] = H[t]
                else:
                    if self.view_aggr == "mean" and count > 0:
                        acc = acc / float(count)
                    H_new[t] = acc
            H = H_new  # next layer input

        # (3) Per-type pooling to sample level, then cross-type fusion
        pooled_per_type: List[torch.Tensor] = []
        # Determine batch size B
        B = len(next(iter(batch_splits.values()))) if len(batch_splits) > 0 else 1

        for t in self.readout_types:
            splits_t = batch_splits.get(t, [0] * B)  # fill zeros if a type never appears, to keep alignment
            pooled_t = self._pool_type_by_splits(H[t], splits_t, mode="mean")
            pooled_per_type.append(pooled_t)  # (B, d)

        G = torch.cat(pooled_per_type, dim=-1) if len(pooled_per_type) > 1 else pooled_per_type[0]  # (B, d*)
        z = self.readout_mlp(G)  # (B, d_model)
        return z









# ---------------------------
# Message Passing Layer (fixed)
# ---------------------------

def _parse_relation_tag_global(tag: str) -> Tuple[str, str]:
    """Return (SRC, DST) from 'A_B' (accepts 'bA_B'/'coA_B' and strips prefix)."""
    t = tag.strip()
    if t.startswith("co"):
        t = t[2:]
    elif t.startswith("b"):
        t = t[1:]
    if "_" not in t:
        raise ValueError(f"Bad relation tag '{tag}'; expected 'A_B' (optionally prefixed by 'b'/'co').")
    a, b = t.split("_", 1)
    return a.upper(), b.upper()


class ChainComplexMessagePassingLayer(nn.Module):
    """
    Single-view, single-layer, hetero multi-relation **one-way** message passing.

    IMPORTANT FIX:
      This layer now allocates outputs per node type using the **global counts from H[t]**,
      not view['sizes'], so that different views produce per-type tensors with **identical row counts**.
    """

    def __init__(self,
                 node_types: List[str],
                 relations: List[str],
                 in_dim: int,
                 out_dim: int,
                 *,
                 num_bases: Optional[int] = 4,
                 aggr: str = "sum",
                 norm: str = "sym",
                 residual: bool = True,
                 self_loop: bool = False,
                 dropout: float = 0.1,
                 act: Optional[nn.Module] = None):
        super().__init__()
        assert aggr in ("sum", "mean")
        assert norm in ("sym", "dst", "none")
        self.node_types = list(node_types)
        self.relations = list(relations)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_bases = None if (num_bases is None) else int(num_bases)
        self.aggr = aggr
        self.norm = norm
        self.residual = residual
        self.self_loop = self_loop
        self.dropout = float(dropout)
        self.act = act if act is not None else nn.GELU()

        # Per-destination-type: LayerNorm (PreNorm), residual projection, update MLP, self-loop gate.
        self.ln_in = nn.ModuleDict({t: nn.LayerNorm(self.in_dim) for t in self.node_types})
        self.res_proj = nn.ModuleDict({t: nn.Linear(self.in_dim, self.out_dim, bias=False)
                                       for t in self.node_types})
        hidden = max(self.out_dim * 2, 64)
        self.update_mlp = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(self.out_dim * 2, hidden, bias=True),
                self.act,
                nn.Dropout(self.dropout),
                nn.Linear(hidden, self.out_dim, bias=True),
            ) for t in self.node_types
        })
        if self.self_loop:
            self.self_gate = nn.ParameterDict({t: nn.Parameter(torch.tensor(1.0)) for t in self.node_types})

        # Relation weights: basis or per-relation
        if self.num_bases is None or self.num_bases >= len(self.relations):
            self.weight_mode = "per_relation"
            self.rel_W = nn.ParameterDict({
                r: nn.Parameter(torch.empty(self.in_dim, self.out_dim)) for r in self.relations
            })
            for p in self.rel_W.values():
                nn.init.xavier_uniform_(p)
        else:
            self.weight_mode = "basis"
            B = self.num_bases
            self.bases = nn.ParameterList([nn.Parameter(torch.empty(self.in_dim, self.out_dim)) for _ in range(B)])
            for V in self.bases:
                nn.init.xavier_uniform_(V)
            self.rel_coeff = nn.ParameterDict({
                r: nn.Parameter(torch.randn(B) / (B ** 0.5)) for r in self.relations
            })

        # Relation gates (scalar)
        self.rel_gate = nn.ParameterDict({r: nn.Parameter(torch.tensor(1.0)) for r in self.relations})
        self.dropout_out = nn.Dropout(self.dropout)

    # ---- helpers ----
    def _get_W(self, r: str) -> torch.Tensor:
        if getattr(self, "weight_mode", "per_relation") == "per_relation":
            return self.rel_W[r]
        # basis
        coeff = self.rel_coeff[r]  # (B,)
        W = torch.zeros((self.in_dim, self.out_dim), device=coeff.device, dtype=self.bases[0].dtype)
        for a, V in zip(coeff, self.bases):
            W = W + a * V
        return W

    @staticmethod
    def _safe_inv_sqrt(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        mask = x > 0
        out[mask] = x[mask].pow(-0.5)
        return out

    @staticmethod
    def _safe_inv(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        mask = x > 0
        out[mask] = x[mask].reciprocal()
        return out

    def _spmm_norm(self,
                   A: torch.Tensor,                # sparse COO (|dst| x |src|)
                   X_src: torch.Tensor,            # (|src| x out_dim)
                   deg_src: torch.Tensor,          # (|src|,)
                   deg_dst: torch.Tensor) -> torch.Tensor:   # (|dst|,)
        """Apply normalization + SpMM for one relation."""
        if self.norm == "none":
            msg = torch.sparse.mm(A, X_src)
            return msg
        elif self.norm == "dst":
            msg = torch.sparse.mm(A, X_src)
            dinv = self._safe_inv(deg_dst).unsqueeze(-1)  # (|dst|,1)
            return msg * dinv
        else:  # 'sym'
            dsrc = self._safe_inv_sqrt(deg_src).unsqueeze(-1)   # (|src|,1)
            Xn = X_src * dsrc
            msg = torch.sparse.mm(A, Xn)
            ddst = self._safe_inv_sqrt(deg_dst).unsqueeze(-1)   # (|dst|,1)
            return msg * ddst

    # ---- forward ----
    def forward(self,
                H: Dict[str, torch.Tensor],
                view: Dict[str, dict]) -> Dict[str, torch.Tensor]:
        """
        H: {type: (N_type_global, in_dim)}
        view: {'adj': {relation_tag: sparse_coo}, 'deg': {relation_tag: (deg_src, deg_dst)}, ...}

        Returns a dict {type: (N_type_global, out_dim)} with GLOBAL row counts
        so that outputs from different views are addable.
        """
        device = next(self.parameters()).device

        # GLOBAL counts from H, not from view['sizes']
        N: Dict[str, int] = {t: (H[t].shape[0] if t in H else 0) for t in self.node_types}

        # Aggregation buffers per destination type (GLOBAL shapes)
        agg_msgs: Dict[str, torch.Tensor] = {
            t: torch.zeros((N.get(t, 0), self.out_dim), device=device) for t in self.node_types
        }
        rel_counts: Dict[str, int] = {t: 0 for t in self.node_types}

        # PreNorm + projection for destination features (GLOBAL shapes)
        proj_dst: Dict[str, torch.Tensor] = {}
        for t in self.node_types:
            n = N.get(t, 0)
            if n == 0:
                proj_dst[t] = torch.zeros((0, self.out_dim), device=device)
                continue
            x = H[t]  # (N_t_global, in_dim)
            x_norm = self.ln_in[t](x)
            proj_dst[t] = self.res_proj[t](x_norm)  # (N_t_global, out_dim)

        # Iterate relations
        for r in self.relations:
            if "adj" not in view or r not in view["adj"]:
                continue  # relation absent
            A = view["adj"][r]                    # (|dst_global| x |src_global|) sparse COO
            deg_src, deg_dst = view["deg"][r]     # 1D tensors
            src, dst = _parse_relation_tag_global(r)

            n_src = N.get(src, 0)
            n_dst = N.get(dst, 0)
            if n_src == 0 or n_dst == 0:
                continue  # no-op

            # Sanity check: adjacency must match global shapes
            assert A.size(0) == n_dst and A.size(1) == n_src, \
                f"Adjacency shape mismatch for relation {r}: got {tuple(A.size())}, expected ({n_dst},{n_src})"

            # 1) Linear map on src
            W_r = self._get_W(r)                  # (in_dim x out_dim)
            X_src = H[src] @ W_r                  # (|src| x out_dim)

            # 2) Normalization + SpMM
            msg = self._spmm_norm(A, X_src, deg_src.to(device), deg_dst.to(device))  # (|dst| x out_dim)

            # 3) Relation gate
            gamma = self.rel_gate[r]
            msg = msg * gamma

            # 4) Accumulate into destination bucket
            agg_msgs[dst] = agg_msgs[dst] + msg
            rel_counts[dst] += 1

        # Optional mean across relations
        if self.aggr == "mean":
            for t in self.node_types:
                cnt = rel_counts[t]
                if cnt > 0:
                    agg_msgs[t] = agg_msgs[t] / float(cnt)

        # Optional self-loop message (independent of residual)
        if self.self_loop:
            for t in self.node_types:
                if N.get(t, 0) == 0:
                    continue
                beta = self.self_gate[t]
                agg_msgs[t] = agg_msgs[t] + beta * proj_dst[t]

        # Update with PreNorm residual MLP
        out: Dict[str, torch.Tensor] = {}
        for t in self.node_types:
            if N.get(t, 0) == 0:
                out[t] = torch.zeros((0, self.out_dim), device=device)
                continue
            upd_in = torch.cat([proj_dst[t], agg_msgs[t]], dim=-1)  # (|t|, 2*out_dim)
            upd = self.update_mlp[t](upd_in)                         # (|t|, out_dim)
            upd = self.dropout_out(upd)
            out[t] = proj_dst[t] + upd if self.residual else upd
        return out


# ---------------------------
# Embedder (fixed)
# ---------------------------

def _cum_offsets(sizes: List[int]) -> List[int]:
    """Compute cumulative offsets for a list of segment sizes."""
    offs = [0]
    s = 0
    for v in sizes:
        s += int(v)
        offs.append(s)
    return offs


class ChainComplexEmbedder(nn.Module):
    """
    Multi-view, multi-relation embedder built from stacks of ChainComplexMessagePassingLayer.

    FIXES:
      • forward(list): packs per-sample views to block-diagonal graphs using **batch_splits** as block sizes.
      • degree vectors are **padded** to block size for each sample.
      • per-type global counts are computed from **batch_splits** (sum) to initialize features.
    """

    def __init__(self,
                 views_info: List[Dict[str, Any]],
                 d_model: int = 128,
                 num_layers: int = 4,
                 *,
                 view_aggr: str = "sum",
                 readout_types: Optional[List[str]] = None,
                 # per-layer options
                 num_bases: Optional[int] = 4,
                 norm: str = "sym",
                 residual: bool = True,
                 self_loop: bool = False,
                 dropout: float = 0.1,
                 act: Optional[nn.Module] = None):
        super().__init__()
        assert view_aggr in ("sum", "mean")
        self.view_aggr = view_aggr
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)

        # Parse schema
        self.views_info = []
        self.view_names: List[str] = []
        self.view_node_types: Dict[str, List[str]] = {}
        self.view_relations: Dict[str, List[str]] = {}
        all_types = set()

        for v in views_info:
            name = v.get("name", "view")
            nts = list(v.get("partite_classes", []))
            rels = list(v.get("relations", []))
            self.views_info.append({"name": name, "partite_classes": nts, "relations": rels})
            self.view_names.append(name)
            self.view_node_types[name] = nts
            self.view_relations[name] = rels
            all_types.update(nts)

        self.all_node_types: List[str] = sorted(all_types)
        self.readout_types: List[str] = list(readout_types) if readout_types is not None else self.all_node_types

        # Type tokens
        self.type_tokens = nn.ParameterDict({
            t: nn.Parameter(torch.empty(1, self.d_model)) for t in self.all_node_types
        })
        for p in self.type_tokens.values():
            nn.init.xavier_uniform_(p)

        # View gates: one scalar per (layer, view)
        self.view_gates = nn.ParameterList([
            nn.ParameterDict({vn: nn.Parameter(torch.tensor(1.0)) for vn in self.view_names})
            for _ in range(self.num_layers)
        ])

        # Per-layer, per-view message passing modules
        Layer = ChainComplexMessagePassingLayer
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            per_view = nn.ModuleDict()
            for vn in self.view_names:
                per_view[vn] = Layer(
                    node_types=self.all_node_types,
                    relations=self.view_relations[vn],
                    in_dim=self.d_model,
                    out_dim=self.d_model,
                    num_bases=num_bases,
                    aggr="sum",
                    norm=norm,
                    residual=residual,
                    self_loop=self_loop,
                    dropout=dropout,
                    act=act if act is not None else nn.GELU(),
                )
            self.layers.append(per_view)

        # Readout MLP
        read_dim = len(self.readout_types) * self.d_model
        hidden = max(read_dim, 128)
        self.readout_mlp = nn.Sequential(
            nn.LayerNorm(read_dim),
            nn.Linear(read_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.d_model),
        )

    # ---------- Helpers ----------

    @staticmethod
    def _parse_relation_tag(tag: str) -> Tuple[str, str]:
        """Return (SRC, DST) from 'A_B' (accepts 'bA_B'/'coA_B' and strips prefix)."""
        t = tag.strip()
        if t.startswith("co"):
            t = t[2:]
        elif t.startswith("b"):
            t = t[1:]
        if "_" not in t:
            raise ValueError(f"Bad relation tag '{tag}'; expected 'A_B' (optionally prefixed by 'b'/'co').")
        a, b = t.split("_", 1)
        return a.upper(), b.upper()

    @staticmethod
    def _concat_sparse_blockdiag(
        parts: List[Optional[torch.Tensor]],
        dst_sizes: List[int],
        src_sizes: List[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build a block-diagonal sparse COO by concatenating sample-wise sparse matrices with row/col offsets.
        Each element in `parts` is a (|dst_i| x |src_i|) sparse COO or None.
        """
        assert len(parts) == len(dst_sizes) == len(src_sizes)
        B = len(parts)

        idx_rows: List[torch.Tensor] = []
        idx_cols: List[torch.Tensor] = []
        vals: List[torch.Tensor] = []

        roff = 0
        coff = 0
        for i in range(B):
            A_i = parts[i]
            if (A_i is not None) and (A_i._nnz() > 0):
                Ai = A_i.coalesce().cpu()
                idx = Ai.indices()
                val = Ai.values().to(dtype=torch.float32)
                idx_rows.append(idx[0] + roff)
                idx_cols.append(idx[1] + coff)
                vals.append(val)
            roff += int(dst_sizes[i])
            coff += int(src_sizes[i])

        total_dst = sum(int(x) for x in dst_sizes)
        total_src = sum(int(x) for x in src_sizes)

        if len(vals) == 0:
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_val = torch.empty((0,), dtype=dtype, device=device)
            return torch.sparse_coo_tensor(empty_idx, empty_val, size=(total_dst, total_src),
                                           device=device, dtype=dtype).coalesce()

        rows = torch.cat(idx_rows, dim=0)
        cols = torch.cat(idx_cols, dim=0)
        vcat = torch.cat(vals, dim=0)

        A = torch.sparse_coo_tensor(
            torch.stack([rows, cols], dim=0),
            vcat,
            size=(total_dst, total_src),
            device=device,
            dtype=dtype,
        ).coalesce()
        return A

    def _pack_batch(
        self,
        sample_views_list: List[Dict[str, Dict[str, Any]]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[int]]]:
        """
        Pack a list of per-sample 'views' into block-diagonal batched graphs for each view.

        Returns
        -------
        packed_views : Dict[view_name -> view_dict]
          Each view_dict contains:
            - 'sizes': {type: N_total}  (sum of per-sample provided sizes; not used for allocation anymore)
            - 'adj'  : {relation_tag: sparse_coo (|DST_global| x |SRC_global|)}
            - 'deg'  : {relation_tag: (deg_src, deg_dst)} with concatenated & padded vectors
        batch_splits : Dict[type -> List[int]]
          Per-type node counts per sample (used to determine global counts and pooling).
        """
        B = len(sample_views_list)

        # (1) Build per-type splits using max across views for robustness
        batch_splits: Dict[str, List[int]] = {t: [0] * B for t in self.all_node_types}
        for b, views in enumerate(sample_views_list):
            for vn in self.view_names:
                vdict = views.get(vn, None)
                if vdict is None:
                    continue
                sz = vdict.get("sizes", {})
                for t in self.all_node_types:
                    batch_splits[t][b] = max(batch_splits[t][b], int(sz.get(t, 0)))

        # (2) Pack each view independently
        packed_views: Dict[str, Dict[str, Any]] = {}
        for vn in self.view_names:
            # We keep 'sizes' as sum of provided sizes for compatibility (not used for allocation)
            totals_v: Dict[str, int] = {t: 0 for t in self.all_node_types}
            for b in range(B):
                vdict = sample_views_list[b].get(vn, None)
                sz = vdict.get("sizes", {}) if vdict is not None else {}
                for t in self.all_node_types:
                    totals_v[t] += int(sz.get(t, 0))

            adj_packed: Dict[str, torch.Tensor] = {}
            deg_packed: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

            for r in self.view_relations[vn]:
                src_t, dst_t = self._parse_relation_tag(r)

                parts: List[Optional[torch.Tensor]] = []
                dst_sizes: List[int] = []
                src_sizes: List[int] = []
                deg_src_list: List[torch.Tensor] = []
                deg_dst_list: List[torch.Tensor] = []

                for b in range(B):
                    vdict = sample_views_list[b].get(vn, None)
                    blk_src = batch_splits[src_t][b]  # ALWAYS use global per-sample block size
                    blk_dst = batch_splits[dst_t][b]

                    # adjacency block (can be None)
                    A = None if vdict is None else vdict.get("adj", {}).get(r, None)
                    parts.append(A if A is not None else None)
                    src_sizes.append(blk_src)
                    dst_sizes.append(blk_dst)

                    # degrees: pad to block size
                    if (vdict is None) or (r not in vdict.get("deg", {})):
                        dsrc = torch.zeros(blk_src, dtype=torch.float32)
                        ddst = torch.zeros(blk_dst, dtype=torch.float32)
                    else:
                        dsrc, ddst = vdict["deg"][r]
                        dsrc = dsrc.detach().cpu()
                        ddst = ddst.detach().cpu()
                        if dsrc.numel() < blk_src:
                            dsrc = torch.cat([dsrc, torch.zeros(blk_src - dsrc.numel(), dtype=torch.float32)], dim=0)
                        elif dsrc.numel() > blk_src:
                            dsrc = dsrc[:blk_src]
                        if ddst.numel() < blk_dst:
                            ddst = torch.cat([ddst, torch.zeros(blk_dst - ddst.numel(), dtype=torch.float32)], dim=0)
                        elif ddst.numel() > blk_dst:
                            ddst = ddst[:blk_dst]

                    deg_src_list.append(dsrc)
                    deg_dst_list.append(ddst)

                # block-diagonal adjacency (GLOBAL shapes)
                A_batch = self._concat_sparse_blockdiag(parts, dst_sizes, src_sizes, device=device, dtype=dtype)

                # concatenated degrees
                deg_src_cat = torch.cat(deg_src_list, dim=0).to(device)
                deg_dst_cat = torch.cat(deg_dst_list, dim=0).to(device)

                adj_packed[r] = A_batch
                deg_packed[r] = (deg_src_cat, deg_dst_cat)

            packed_views[vn] = {
                "sizes": totals_v,   # kept for reference; layer does not rely on it for allocation
                "adj": adj_packed,
                "deg": deg_packed,
            }

        return packed_views, batch_splits

    @staticmethod
    def _build_initial_H_from_splits(batch_splits: Dict[str, List[int]],
                                     type_tokens: nn.ParameterDict,
                                     device: torch.device) -> Dict[str, torch.Tensor]:
        """Create initial features by repeating per-type tokens; counts from sum of batch_splits."""
        H: Dict[str, torch.Tensor] = {}
        for t, splits in batch_splits.items():
            n = int(sum(int(x) for x in splits))
            if n <= 0:
                H[t] = torch.zeros((0, type_tokens[t].shape[1]), device=device)
            else:
                H[t] = type_tokens[t].to(device).expand(n, -1).clone()
        return H

    @staticmethod
    def _pool_type_by_splits(H_t: torch.Tensor, splits: Optional[List[int]], mode: str = "mean") -> torch.Tensor:
        """Per-type, per-sample pooling for a batched graph."""
        if splits is None:
            if H_t.numel() == 0:
                return H_t.new_zeros((1, H_t.shape[1]))
            return H_t.mean(dim=0, keepdim=True) if mode == "mean" else H_t.sum(dim=0, keepdim=True)

        B = len(splits)
        offs = _cum_offsets(splits)
        out = H_t.new_zeros((B, H_t.shape[1]))
        for b in range(B):
            s, e = offs[b], offs[b + 1]
            if e > s:
                out[b] = H_t[s:e].mean(dim=0) if mode == "mean" else H_t[s:e].sum(dim=0)
            else:
                out[b] = 0.0
        return out

    # ---------- Forward ----------

    def forward(self,
                views_or_list: Any,
                batch_splits: Optional[Dict[str, List[int]]] = None) -> torch.Tensor:
        """
        Accept either:
          • a list of per-sample views (output of RelationEncoder.encode for each sample), or
          • a packed dict of views (block-diagonal batched graphs).
        Returns:
          Tensor of shape (B, d_model).
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # If a list is provided, pack it first
        if isinstance(views_or_list, list):
            packed_views, batch_splits = self._pack_batch(views_or_list, device=device, dtype=dtype)
            views = packed_views
            # Build initial features using global counts from batch_splits
            H = self._build_initial_H_from_splits(batch_splits, self.type_tokens, device=device)
        else:
            views = views_or_list
            # If no batch_splits are provided, assume B=1 and infer counts from views (best effort)
            if batch_splits is None:
                totals_single: Dict[str, int] = {}
                for t in self.all_node_types:
                    totals_single[t] = 0
                    for v in views.values():
                        totals_single[t] = max(totals_single[t], int(v.get("sizes", {}).get(t, 0)))
                batch_splits = {t: [totals_single.get(t, 0)] for t in self.all_node_types}
            H = self._build_initial_H_from_splits(batch_splits, self.type_tokens, device=device)

        # L stacked layers; per layer: run every view, then aggregate across views
        for li, per_view in enumerate(self.layers):
            outs_per_view: Dict[str, Dict[str, torch.Tensor]] = {}
            for vn in self.view_names:
                if vn not in views:
                    continue
                outs_per_view[vn] = per_view[vn](H, views[vn])

            # Cross-view aggregation -> new H (shapes are identical across views per type)
            H_new: Dict[str, torch.Tensor] = {}
            for t in self.all_node_types:
                acc = None
                count = 0
                for vn, out_v in outs_per_view.items():
                    if t in out_v:
                        w = self.view_gates[li][vn]
                        contrib = out_v[t] * w
                        acc = contrib if acc is None else (acc + contrib)
                        count += 1
                if acc is None:
                    H_new[t] = H[t]
                else:
                    if self.view_aggr == "mean" and count > 0:
                        acc = acc / float(count)
                    H_new[t] = acc
            H = H_new  # next layer input

        # Per-type pooling to sample level, then cross-type fusion
        pooled_per_type: List[torch.Tensor] = []
        B = len(next(iter(batch_splits.values()))) if len(batch_splits) > 0 else 1

        for t in self.readout_types:
            splits_t = batch_splits.get(t, [0] * B)  # fill zeros if a type never appears
            pooled_t = self._pool_type_by_splits(H[t], splits_t, mode="mean")
            pooled_per_type.append(pooled_t)  # (B, d)

        G = torch.cat(pooled_per_type, dim=-1) if len(pooled_per_type) > 1 else pooled_per_type[0]  # (B, d*)
        z = self.readout_mlp(G)  # (B, d_model)
        return z

"""
    This file defines encoder that converts the input code into different representations
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from toponetx import CombinatorialComplex
from torch_geometric.utils import dense_to_sparse
from typing import List
from scipy.sparse import csr_matrix

import numpy as np
from torch_geometric.data import Data, DataLoader
from bayesian_optimization.gadget import *


class CSSEncoder():
    def __init__(self,info,mode='graph',grakel_use = False,logical=False,cc_patern=[]):
        """
            n: number of qubits, n = hx.shape[1] = hz.shape[1]
            nx: number of x stabilizers, nx = hx.shape[0]
            nz: number of z stabilizers, nz = hz.shape[0]
            mode: 'graph' or 'combinatorial_complex'
        """
        self.logical = logical
        self.mode=mode
        if mode == 'graph':
            n = info['n']
            nx = info['nx']
            nz = info['nz']
            self.encoder = GraphEncoder(n,nx,nz,grakel_use=grakel_use,logical=logical)
        elif mode == 'combinatorial_complex':
            n = info['n']
            nx = info['nx']
            nz = info['nz']
            self.encoder = CombinatorialComplexEncoder_(n,nx,nz)
        elif mode == 'chain_complex':
            self.encoder = ChainComplexGlobalFeaturesEncoder(info)
        elif mode == 'relations':
            self.encoder = RelationEncoder(info)
    def encode(self,c):
        if self.mode in ['chain_complex','relations']:
            return self.encoder.encode(c) 
        if self.logical:
            hx = c.hx
            hz = c.hz
            lx = c.lx
            lz = c.lz
            return self.encoder.encode(hx,hz,lx,lz)
        else:
            hx = c.hx
            hz = c.hz
            return self.encoder.encode(hx,hz)
    




class GraphEncoder():
    def __init__(self,n,nx,nz,grakel_use=False,logical=False):
        """
            n: number of qubits, n = hx.shape[1] = hz.shape[1]
            nx: number of x stabilizers, nx = hx.shape[0]
            nz: number of z stabilizers, nz = hz.shape[0]
        """
        self.n = n
        self.nx = nx
        self.nz = nz
        self.grakel_use = grakel_use
        if grakel_use:
            from grakel import Graph
        self.logical = logical
    def preprocess_graph(self,adj_matrix: np.ndarray, node_onehots: np.ndarray):
        """
        adj_matrix: np.ndarray of shape (N, N)
        node_onehots: np.ndarray of shape (N, 5)
        """
  
        src, dst = np.nonzero(adj_matrix)
        edge_index = torch.tensor([src, dst], dtype=torch.long)  # shape [2, E]


        x = torch.tensor(node_onehots, dtype=torch.float)         # shape [N, 5]

  
        data = Data(x=x, edge_index=edge_index)
        return data
        
    def encode(self,hx,hz,lx=None,lz=None):
        
                    # adjacency_matrix[self.n+self.nx+j][i] = 1
        # print(f'adjacency_matrix:{adjacency_matrix}')
        # edge_index = dense_to_sparse(adjacency_matrix)[0]
        # return edge_index
        if self.logical==False:
            adjacency_matrix = np.zeros((self.n+self.nx+self.nz,self.n+self.nx+self.nz))
            
            for i in range(self.n):
                for j in range(self.nx):
                    if hx[j][i] == 1:
                        adjacency_matrix[i][self.n+j] = 1
                        adjacency_matrix[self.n+j][i] = 1
                for j in range(self.nz):
                    if hz[j][i] == 1:
                        adjacency_matrix[i][self.n+self.nx+j] = 1
                        adjacency_matrix[self.n+self.nx+j][i] = 1
            if not self.grakel_use:
                
                return torch.tensor(adjacency_matrix)
            else:
                
                
                node_labels = {}
                for i in range(self.n):
                    node_labels[i]='0'
                for i in range(self.n,self.n+self.nx):
                    node_labels[i]='+'
                for i in range(self.n+self.nx,self.n+self.nx+self.nz):
                    node_labels[i]='-'
                
                # print('adjacency_matrix')
                # print(adjacency_matrix)
                g = Graph(initialization_object=adjacency_matrix,node_labels=node_labels,graph_format='adjacency',construct_labels=False)
                # print(g)
                return g
        else:
            total_nodes = self.n+self.nx+self.nz+lx.shape[0]+lz.shape[0]
            adjacency_matrix = np.zeros((total_nodes,total_nodes))
            for i in range(self.n):
                for j in range(self.nx):
                    if hx[j][i] == 1:
                        adjacency_matrix[i][self.n+j] = 1
                        adjacency_matrix[self.n+j][i] = 1
                for j in range(self.nz):
                    if hz[j][i] == 1:
                        adjacency_matrix[i][self.n+self.nx+j] = 1
                        adjacency_matrix[self.n+self.nx+j][i] = 1
                for j in range(lx.shape[0]):
                    if lx[j,i]==1:
                        adjacency_matrix[i][self.n+self.nx+self.nz+j]=1
                        adjacency_matrix[self.n+self.nx+self.nz+j][i]=1
                for j in range(lz.shape[0]):
                    if lz[j,i]==1:
                        adjacency_matrix[i][self.n+self.nx+self.nz+lx.shape[0]+j]=1
                        adjacency_matrix[self.n+self.nx+self.nz+lx.shape[0]+j][i]=1
            if not self.grakel_use:
                node_onehots = torch.zeros((total_nodes, 5), dtype=torch.float)
                node_onehots[0        : self.n,    0] = 1.0
                node_onehots[self.n    : self.n + self.nx,        1] = 1.0
                node_onehots[self.n + self.nx : self.n + self.nx + self.nz, 2] = 1.0
                node_onehots[self.n + self.nx + self.nz : self.n + self.nx + self.nz + lx.shape[0], 3] = 1.0
                node_onehots[self.n + self.nx + self.nz + lx.shape[0] : total_nodes, 4] = 1.0
                return self.preprocess_graph(adjacency_matrix,node_onehots)
            else:
                
                
                node_labels = {}
                for i in range(self.n):
                    node_labels[i]='0'
                for i in range(self.n,self.n+self.nx):
                    node_labels[i]='+'
                for i in range(self.n+self.nx,self.n+self.nx+self.nz):
                    node_labels[i]='-'
                for i in range(self.n+self.nx+self.nz,self.n+self.nx+self.nz+lx.shape[0]):
                    node_labels[i]='x'
                for i in range(self.n+self.nx+self.nz+lx.shape[0],self.n+self.nx+self.nz+lx.shape[0]+lz.shape[0]):
                    node_labels[i]='z'
                
                # print('adjacency_matrix')
                # print(adjacency_matrix)
                g = Graph(initialization_object=adjacency_matrix,node_labels=node_labels,graph_format='adjacency',construct_labels=False)
                # print(g)
                return g
        
        
        

    


# optional SciPy sparse backend
try:
    import scipy.sparse as sp
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _to_bool_int(a: np.ndarray) -> np.ndarray:
    """Map any ndarray to {0,1} as int64, no copy when possible."""
    if a.dtype == np.bool_:
        return a.astype(np.int64, copy=False)
    if np.issubdtype(a.dtype, np.integer):
        return (a & 1).astype(np.int64, copy=False)
    return (a != 0).astype(np.int64, copy=False)


def _vstack_bool_int(*rows: np.ndarray) -> np.ndarray:
    """vstack rows after binarization to {0,1} int64."""
    rows_b = [_to_bool_int(r) for r in rows if r is not None and r.size > 0]
    if not rows_b:
        return np.zeros((0, rows[0].shape[1]), dtype=np.int64)  # assume widths match if any
    return np.vstack(rows_b).astype(np.int64, copy=False)





def _to_bool_int(a: np.ndarray) -> np.ndarray:
    """Map any ndarray to {0,1} int64 (GF(2) semantics)."""
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.integer):
        return (a & 1).astype(np.int64, copy=False)
    return (a != 0).astype(np.int64, copy=False)


def _vstack_bool_int(*rows: np.ndarray) -> np.ndarray:
    """vstack rows after GF(2) binarization to {0,1} int64."""
    rows_b = [_to_bool_int(r) for r in rows if r is not None and r.size > 0]
    if not rows_b:
        # width fallback: if nothing provided, we cannot infer width safely
        return np.zeros((0, 0), dtype=np.int64)
    width = rows_b[0].shape[1]
    for rb in rows_b:
        assert rb.shape[1] == width, "All stacked matrices must have the same number of columns"
    return np.vstack(rows_b).astype(np.int64, copy=False)


def _np_to_sparse_coo(M: np.ndarray,
                      device: Optional[torch.device],
                      dtype: torch.dtype) -> torch.Tensor:
    """
    Convert 0/1 numpy array to torch.sparse_coo_tensor of shape M.shape on device.
    Values are 1.0 (dtype provided). Always coalesced.
    """
    M = np.asarray(M)
    m, n = M.shape
    rows, cols = np.nonzero(M)
    nnz = rows.size
    if nnz == 0:
        idx = torch.empty((2, 0), dtype=torch.long, device=device)
        val = torch.empty((0,), dtype=dtype, device=device)
        return torch.sparse_coo_tensor(idx, val, size=(m, n), device=device, dtype=dtype).coalesce()
    idx = torch.from_numpy(np.vstack([rows, cols])).long().to(device)
    val = torch.ones((nnz,), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, size=(m, n), device=device, dtype=dtype).coalesce()


class RelationEncoder:
    """
    Build relation matrices (torch.sparse_coo_tensor) for specified graph 'views'.

    Conventions
    -----------
    - Node classes (7):
        'SX' (rows of Hx), 'SZ' (rows of Hz), 'DQ' (n),
        'LX' (rows of Lx), 'LZ' (rows of Lz),
        'AX' = vstack(Hx, Lx), 'AZ' = vstack(Hz, Lz).
    - Relation tag 'A_B' means messages flow A -> B,
      and the matrix has shape (|B|, |A|), i.e. left-multiply to aggregate into B:
          H_B += A_B @ (H_A @ W)
    - Prefix:
        'bA_B'  : alias of 'A_B'
        'coA_B' : transpose direction (B -> A), i.e. matrix = (A_B).T
        'adj*'  : reserved (not implemented yet)

    views_info (per view)
    ---------------------
      {
        "name": str,
        "partite_classes": List[str],               # subset of {"SX","SZ","DQ","LX","LZ","AX","AZ"}
        "relations": List[str],                     # e.g. ["SZ_DQ","DQ_SX","AX_DQ", ...]
      }

    Output (per view)
    -----------------
      {
        "sizes": Dict[node_type -> int],
        "adj":   Dict[relation_tag -> torch.sparse_coo_tensor],    # (|DST| x |SRC|)
        "deg":   Dict[relation_tag -> (deg_src: torch.Tensor, deg_dst: torch.Tensor)],
                # deg_src shape (|SRC|,), deg_dst shape (|DST|,), dtype float32, same device
      }
    """

    NODE_SET = {"SX", "SZ", "DQ", "LX", "LZ", "AX", "AZ"}

    def __init__(self,
                 views_info: Optional[List[Dict[str, Any]]] = None,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32):
        self.views_info = views_info or []
        self.device = device
        self.dtype = dtype

    # ---------- core API ----------

    def encode(self, c) -> Dict[str, Dict[str, Any]]:
        """
        Build all requested views for CSSCode 'c' which has:
            c.n (int)
            c.hx, c.hz, c.lx, c.lz (np.ndarray, shapes [m_x,n], [m_z,n], [k_x,n], [k_z,n])
        All matrices are treated over GF(2) and converted to {0,1}.
        """
        # Base matrices (GF(2) -> {0,1})
        Hx = _to_bool_int(np.asarray(c.hx))
        Hz = _to_bool_int(np.asarray(c.hz))
        Lx = _to_bool_int(np.asarray(c.lx))
        Lz = _to_bool_int(np.asarray(c.lz))

        n = int(c.n)
        mx = Hx.shape[0]
        mz = Hz.shape[0]
        kx = Lx.shape[0]
        kz = Lz.shape[0]

        AX = _vstack_bool_int(Hx, Lx)  # rows = mx + kx
        AZ = _vstack_bool_int(Hz, Lz)  # rows = mz + kz

        # Sanity on widths
        assert Hx.shape[1] == n and Hz.shape[1] == n and Lx.shape[1] == n and Lz.shape[1] == n, \
            "All Hx/Hz/Lx/Lz must have width n"

        # Sizes per node type
        sizes_all = {
            "SX": mx, "SZ": mz, "DQ": n,
            "LX": kx, "LZ": kz,
            "AX": AX.shape[0], "AZ": AZ.shape[0],
        }

        # Primitive matrices by (SRC, DST) with our left-multiply convention
        # Shape is (|DST|, |SRC|)
        prim_np: Dict[Tuple[str, str], np.ndarray] = {
            ("SX", "DQ"): Hx.T,    ("DQ", "SX"): Hx,
            ("SZ", "DQ"): Hz.T,    ("DQ", "SZ"): Hz,
            ("LX", "DQ"): Lx.T,    ("DQ", "LX"): Lx,
            ("LZ", "DQ"): Lz.T,    ("DQ", "LZ"): Lz,
            ("AX", "DQ"): AX.T,    ("DQ", "AX"): AX,
            ("AZ", "DQ"): AZ.T,    ("DQ", "AZ"): AZ,
        }

        out: Dict[str, Dict[str, Any]] = {}

        for view in self.views_info:
            name = view.get("name", "view")
            partite: List[str] = list(view.get("partite_classes", []))
            relations: List[str] = list(view.get("relations", []))

            self._validate_partite(partite)
            sizes = {t: sizes_all[t] for t in partite}

            adj_dict: Dict[str, torch.Tensor] = {}
            deg_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

            for tag in relations:
                mat_np, (src, dst) = self._build_relation_matrix(tag, prim_np)

                # skip relations whose endpoints are not in this view
                if (src not in partite) or (dst not in partite):
                    continue

                # Convert to torch.sparse_coo (0/1, no multi-edges by assumption)
                A = _np_to_sparse_coo(mat_np, device=self.device, dtype=self.dtype)

                # Degrees (float32 tensors on same device)
                deg_dst_np = mat_np.sum(axis=1).astype(np.float32, copy=False)
                deg_src_np = mat_np.sum(axis=0).astype(np.float32, copy=False)
                deg_dst = torch.from_numpy(deg_dst_np).to(self.device)
                deg_src = torch.from_numpy(deg_src_np).to(self.device)

                adj_dict[tag] = A
                deg_dict[tag] = (deg_src, deg_dst)

            out[name] = {"sizes": sizes, "adj": adj_dict, "deg": deg_dict}

        return out

    # ---------- helpers ----------

    def _validate_partite(self, partite: List[str]) -> None:
        unknown = set(partite) - self.NODE_SET
        if unknown:
            raise ValueError(f"Unknown node classes in view: {sorted(unknown)}")

        # if ("LX" in partite and "SX" in partite) or ("LZ" in partite and "SZ" in partite):
        #     pass
        # if ("LX" in partite and "AX" in partite) or ("LZ" in partite and "AZ" in partite):
        #     pass

    def _build_relation_matrix(self, tag: str,
                               prim_np: Dict[Tuple[str, str], np.ndarray]) -> Tuple[np.ndarray, Tuple[str, str]]:
        """
        Return (numpy matrix, (src,dst)) for relation tag.

        Supported now:
          - 'A_B'    : direct primitive if available
          - 'bA_B'   : alias of 'A_B'
          - 'coA_B'  : transpose of 'A_B' (i.e., B->A)
        Reserved:
          - 'adj*'   : not implemented yet
        """
        raw = tag.strip()

        if raw.startswith("adj"):
            raise NotImplementedError(f"Relation '{tag}': 'adj*' builders are reserved for later.")

        if raw.startswith("co"):
            core = raw[2:]
            src, dst = self._parse_A_B(core)
            base = self._lookup_primitive(src, dst, prim_np, tag)
            return base.T, (dst, src)

        if raw.startswith("b"):
            core = raw[1:]
            src, dst = self._parse_A_B(core)
            base = self._lookup_primitive(src, dst, prim_np, tag)
            return base, (src, dst)

        # plain A_B
        src, dst = self._parse_A_B(raw)
        base = self._lookup_primitive(src, dst, prim_np, tag)
        return base, (src, dst)

    @staticmethod
    def _parse_A_B(core: str) -> Tuple[str, str]:
        if "_" not in core:
            raise ValueError(f"Relation tag must be 'A_B', got '{core}'")
        a, b = core.split("_", 1)
        return a.strip().upper(), b.strip().upper()

    @staticmethod
    def _lookup_primitive(src: str, dst: str,
                          prim_np: Dict[Tuple[str, str], np.ndarray],
                          tag: str) -> np.ndarray:
        key = (src, dst)
        if key not in prim_np:
            raise ValueError(f"Relation '{tag}': unsupported primitive mapping '{src}->{dst}'.")
        return prim_np[key]


        

class CombinatorialComplexEncoder_():
    def __init__(self,n,nx,nz):
        """
            n: number of qubits, n = hx.shape[1] = hz.shape[1]
            nx: number of x stabilizers, nx = hx.shape[0]
            nz: number of z stabilizers, nz = hz.shape[0]
        """
        self.n_c0 = nx
        self.n_c1 = n
        self.n_c2 = nz
        
    def encode(self,hx,hz) -> List[csr_matrix]:
        """
            return Tensor:
            tensor([coA01,coA02,A10,coA12,A20,A21,B01,B02,B12])
        """
        CSS_cc = CombinatorialComplex()
        incidence_01 = [[] for _ in range(self.n_c1)]
        incidence_02 = [[] for _ in range(self.n_c2)]

        for i in range(self.n_c0): 
            for j in range(self.n_c1):  
                if hx[i][j] == 1:
                    incidence_01[j].append(i)

    
        for i in range(self.n_c2):  
            for j in range(self.n_c1):  
                if hz[i][j] == 1:
                    incidence_02[i].append(j) 

   
        for i in range(self.n_c1):
            CSS_cc.add_cell(incidence_01[i], rank=1)
        for i in range(self.n_c2):
            CSS_cc.add_cell(incidence_02[i], rank=2)
        
        return [
            torch.tensor(CSS_cc.coadjacency_matrix(1,0).todense()),
            torch.tensor(CSS_cc.coadjacency_matrix(2,0).todense()),
            torch.tensor(CSS_cc.adjacency_matrix(1,0).todense()),
            torch.tensor(CSS_cc.coadjacency_matrix(2,1).todense()),
            torch.tensor(CSS_cc.adjacency_matrix(2,0).todense()),
            torch.tensor(CSS_cc.adjacency_matrix(2,1).todense()),
            torch.tensor(CSS_cc.boundary_matrix(0,1).todense()),
            torch.tensor(CSS_cc.boundary_matrix(0,2).todense()),
            torch.tensor(CSS_cc.boundary_matrix(1,2).todense())
        ]


if __name__ == '__main__':
    
    Hx = [[1,1,1,0,0,0,1,0],[0,0,1,0,1,1,1,0],[1,1,0,1,0,0,0,1],[0,0,0,1,1,1,0,1]]
    Hz = [[0,1,0,0,1,0,1,1],[0,1,1,1,1,0,0,0],[1,0,0,0,0,1,1,1],[1,0,1,1,0,1,0,0]]
    encoder = CSSEncoder(8,4,4,mode='graph',grakel_use=True)
    class c():
        def __init__(self,hx,hz):
            self.hx = hx
            self.hz=hz
    relations = encoder.encode(c(hx= Hx,hz= Hz))
    print(type(relations))
    # print(relations.dictionary)
    # print(relations)
    # print(type(relations[0]))


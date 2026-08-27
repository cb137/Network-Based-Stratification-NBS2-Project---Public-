import os
import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, csc_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_network(file_name, output_dir='', add_selfloop=True):
    print("* Loading network...")
    df = pd.read_table(file_name)
    nfeatures = len(df.columns) - 2

    if add_selfloop:
        df["self_loop"] = 0.0
    df["intercept"] = 1.0

    node_set = set(df.iloc[:, 0]) | set(df.iloc[:, 1])
    nodes = sorted(list(node_set))
    node2index = {node: i for i, node in enumerate(nodes)}

    if add_selfloop:
        selfloop_list = []
        for node in nodes:
            selfloop_list.append(
                [node, node] + [0.0] * nfeatures + [1.0, 1.0]
            )
        selfloop_df = pd.DataFrame(selfloop_list, columns=df.columns)
        df = pd.concat([df, selfloop_df], ignore_index=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "index_nodes"), "w") as f:
            for i, node in enumerate(nodes):
                f.write(f"{i}\t{node}\n")

    # Map node IDs to indices
    edges = (
        df.iloc[:, :2]
        .applymap(node2index.get)
        .to_numpy(dtype=np.int64)
    )
    # Edge features as sparse CSC
    features = csc_matrix(df.iloc[:, 2:].to_numpy(dtype=float))

    return edges, features, nodes


def load_samples(file_name, nodes, output_dir=''):
    df = pd.read_table(file_name, index_col=0)
    samples = list(df.index)

    # Reindex columns to full node list (missing entries -> 0)
    P_init_df = pd.DataFrame(index=df.index, columns=nodes)
    P_init_df.update(df)
    P_init_df = P_init_df.fillna(0.0)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "index_samples"), "w") as f:
            for i, s in enumerate(samples):
                f.write(f"{i}\t{s}\n")

    P_init = csr_matrix(P_init_df.to_numpy(dtype=float))
    return P_init, samples


def load_grouplabels(file_name):
    group_labels = []
    with open(file_name) as f:
        for line in f.read().rstrip().splitlines():
            row = line.split("\t")
            group_labels.append(row[1])
    return group_labels


def build_group_mappings(group_labels):
    groups = sorted(list(set(group_labels)))
    label2idx = {g: i for i, g in enumerate(groups)}

    sample2group = torch.tensor(
        [label2idx[g] for g in group_labels],
        dtype=torch.long
    )
    G = len(groups)
    M = len(group_labels)
    group_sizes = torch.zeros(G, dtype=torch.float64)
    for i in range(M):
        group_sizes[sample2group[i]] += 1.0

    return sample2group, group_sizes, groups


def csr_to_dense_torch(P_csr, device, renorm_rows=True):
    arr = P_csr.toarray().astype(np.float64)
    if renorm_rows:
        row_sum = arr.sum(axis=1, keepdims=True)
        row_sum = row_sum + 1e-8
        arr = arr / row_sum
    return torch.from_numpy(arr).to(device)


# ============================================================
# 2. Core SRW model
# ============================================================

class SRW_solver(nn.Module):
    """
    Supervised Random Walk (NBS²-style).

    Equations (4)–(6):

        J(w) = λ ||w||_1 + Σ_u 1 / (1 + exp(-β D_u))

        D_u = ||p_u - c_a||^2 - min_{b≠a} ||p_u - c_b||^2

    with:
      - leave-one-out centroids for train (Eq. 6)
      - centroids computed on training set and reused for validation
      - L1 regularization on w (excluding the intercept feature)
    """

    def __init__(
        self,
        edges,
        edge_features,
        P_init_train,
        group_labels_train,
        rst_prob,
        lam=1e-3,
        norm_type="L1",
        WMW_b=2e-4,
        P_init_val=None,
        group_labels_val=None,
        device=None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = torch.float64

        # Graph structure
        self.edges = torch.from_numpy(edges).long().to(device)       # [E, 2]
        self.nnodes = int(self.edges.max().item()) + 1

        # Edge features (convert csc/csr -> dense)
        if isinstance(edge_features, (csr_matrix, csc_matrix)):
            edge_features = edge_features.toarray()
        self.edge_features = torch.from_numpy(
            edge_features.astype(np.float64)
        ).to(device)
        self.nfeatures = self.edge_features.shape[1]

        # Learnable edge weights (small Gaussian)
        self.w = nn.Parameter(
            0.01 * torch.randn(self.nfeatures, device=device, dtype=self.dtype)
        )

        # Random walk + regularization hyperparams
        self.rst_prob = float(rst_prob)
        self.lam = float(lam)
        self.norm_type = norm_type.upper()
        self.WMW_b = float(WMW_b)

        # Training data in P-space (row-normalized)
        self.P_init_train = P_init_train.to(device=device, dtype=self.dtype)  # [M_train, N]
        self.M_train = self.P_init_train.shape[0]

        self.sample2group_train, self.group_sizes_train, self.groups_train = \
            build_group_mappings(group_labels_train)
        self.sample2group_train = self.sample2group_train.to(device)
        self.group_sizes_train = self.group_sizes_train.to(device=self.device, dtype=self.dtype)
        self.G_train = len(self.groups_train)

        # Validation data (row-normalized)
        if P_init_val is not None and group_labels_val is not None:
            self.P_init_val = P_init_val.to(device=device, dtype=self.dtype)
            self.M_val = self.P_init_val.shape[0]
            self.sample2group_val, self.group_sizes_val, self.groups_val = \
                build_group_mappings(group_labels_val)
            self.sample2group_val = self.sample2group_val.to(device)
            self.group_sizes_val = self.group_sizes_val.to(device=self.device, dtype=self.dtype)
            self.G_val = len(self.groups_val)
        else:
            self.P_init_val = None
            self.sample2group_val = None
            self.group_sizes_val = None
            self.M_val = 0
            self.G_val = 0

    # -----------------------------
    # Q(w): transition matrix
    # -----------------------------

    def compute_Q(self):
        N = self.nnodes
        X = self.edge_features  # [E, F]

        # Edge strength via logistic
        z = X @ self.w              # [E]
        strengths = torch.sigmoid(z)  # [E]

        # Build dense strength matrix S
        S = torch.zeros((N, N), dtype=self.dtype, device=self.device)
        src = self.edges[:, 0]
        dst = self.edges[:, 1]
        S[src, dst] = strengths

        # Row-normalize (renorm)
        row_sum = S.sum(dim=1, keepdim=True)
        row_sum = row_sum + 1e-8
        Q = S / row_sum

        return Q  # [N, N]

    # -----------------------------
    # P(w): Personalized PageRank
    # -----------------------------

    def compute_P_train_and_val(self, Q):
        """
        Closed-form solution for Personalized PageRank (P).

        iteration:
            P_new = (1 - r) P Q + r P0

        At convergence:
            P = r P0 (I - (1 - r) Q)^(-1)

        We solve:
            P (I - (1 - r) Q) = r P0
        =>  (I - (1 - r) Q)^T X = (r P0)^T  and X^T = P
        """
        alpha = self.rst_prob
        N = self.nnodes
        I = torch.eye(N, device=self.device, dtype=self.dtype)

        A = I - (1.0 - alpha) * Q        # [N, N]
        B = A.T                        

        rhs_list = []
        rhs_list.append((alpha * self.P_init_train).T)  # [N, M_train]

        if self.P_init_val is not None:
            rhs_list.append((alpha * self.P_init_val).T)  # [N, M_val]

        rhs = torch.cat(rhs_list, dim=1)  # [N, M_total]

        sol = torch.linalg.solve(B, rhs)  # [N, M_total]
        P_all = sol.T                     # [M_total, N]

        P_train = P_all[:self.M_train]

        if self.P_init_val is not None:
            P_val = P_all[self.M_train:]
        else:
            P_val = None

        return P_train, P_val

    # -----------------------------
    # Centroids in P-space
    # -----------------------------

    def compute_centroids(self, P, sample2group, group_sizes):
        """
        Compute group centroids C in P-space.

        P : [M, N]
        sample2group : [M]
        group_sizes  : [G]

        Returns
        -------
        C : [G, N]
        """
        M, N = P.shape
        G = group_sizes.shape[0]

        one_hot = F.one_hot(sample2group, num_classes=G).to(self.dtype)  # [M, G]
        group_sums = one_hot.T @ P                                       # [G, N]
        counts = group_sizes.view(-1, 1)                                 # [G, 1]
        C = group_sums / (counts + 1e-8)
        return C

    # -----------------------------
    # WMW loss for training (Eq. 4–6 with leave-one-out centroid)
    # -----------------------------

    def wmw_loss_train(self, P, sample2group, group_sizes, WMW_b):
        """
        Training-set WMW cost using paper’s D_u.

        For each sample u (true subtype a):

            D_u = ||(m_a/(m_a-1)) (p_u - c_a)||^2   -   min_{b≠a} ||p_u - c_b||^2
        """
        device = self.device
        M, N = P.shape
        G = int(group_sizes.shape[0])

        # Centroids on training set
        C = self.compute_centroids(P, sample2group, group_sizes)  # [G, N]

        # Precompute norms & dot products
        P_norm2 = (P ** 2).sum(dim=1)      # [M]
        C_norm2 = (C ** 2).sum(dim=1)      # [G]
        dot_PC = P @ C.T                   # [M, G]

        # Distances ||p_u - c_j||^2, shape [M, G]
        dist_all = P_norm2[:, None] - 2.0 * dot_PC + C_norm2[None, :]

        group_idx = sample2group          # [M]

        # Own-cluster distance: ||p_u - c_a||^2
        dist_ui_raw = dist_all[torch.arange(M, device=device), group_idx]  # [M]
        m_i = group_sizes[group_idx]      # [M]

        # Leave-one-out factor (m_a/(m_a-1))^2, avoiding div-by-zero
        frac = torch.ones_like(m_i)
        mask = m_i > 1.0
        frac[mask] = m_i[mask] / (m_i[mask] - 1.0)
        frac_sq = frac ** 2
        dist_ui = frac_sq * dist_ui_raw   # [M]

        # Distances to other clusters only
        mask_self = F.one_hot(group_idx, num_classes=G).bool().to(device)
        dist_others = dist_all.masked_fill(mask_self, 1e9) 

        # min_{b≠a} ||p_u - c_b||^2
        dist_wrong_min, _ = dist_others.min(dim=1)  # [M]

        # D_u = own_dist_LOO - min_other_dist
        D_u = dist_ui - dist_wrong_min             # [M]

        # Accuracy: correct if D_u < 0
        accuracy = (D_u < 0).double().mean()

        # WMW logistic cost: cost_u = 1 / (1 + exp(-D_u / b))
        cost_u = torch.sigmoid(D_u / WMW_b)        # [M]
        cost = cost_u.sum()

        return cost, accuracy, C

    # -----------------------------
    # WMW loss for validation (same D_u, centroids from train)
    # -----------------------------

    def wmw_loss_val(self, P_val, C_train, sample2group_val, WMW_b):
        """
        Validation-set WMW loss using the same D_u formula as paper,
        with centroids C computed from training set only.
        """
        if P_val is None:
            return None, None

        device = self.device
        M, N = P_val.shape
        G = C_train.shape[0]

        # Norms and dot products
        P_norm2 = (P_val ** 2).sum(dim=1)        # [M]
        C_norm2 = (C_train ** 2).sum(dim=1)      # [G]
        dot_PC = P_val @ C_train.T               # [M, G]

        dist_all = P_norm2[:, None] - 2.0 * dot_PC + C_norm2[None, :]

        group_idx = sample2group_val             # [M]
        dist_ui = dist_all[torch.arange(M, device=device), group_idx]  # [M]

        mask_self = F.one_hot(group_idx, num_classes=G).bool().to(device)
        dist_others = dist_all.masked_fill(mask_self, 1e9)
        dist_wrong_min, _ = dist_others.min(dim=1)

        D_u = dist_ui - dist_wrong_min           # [M]

        accuracy = (D_u < 0).double().mean()
        cost_u = torch.sigmoid(D_u / WMW_b)
        cost = cost_u.sum()

        return cost, accuracy

    # -----------------------------
    # Regularization on w
    # -----------------------------

    def regularization(self):
        w = self.w
        if self.nfeatures > 1:
            w_main = w[:-1]
        else:
            w_main = w

        if self.norm_type == "L2":
            reg = (w_main ** 2).sum()
        elif self.norm_type == "L1":
            reg = w_main.abs().sum()
        else:
            reg = torch.zeros((), device=self.device, dtype=self.dtype)

        return reg

    # -----------------------------
    # One full forward pass
    # -----------------------------

    def forward(self):
        Q = self.compute_Q()
        P_train, P_val = self.compute_P_train_and_val(Q)

        # Training loss (with leave-one-out scaling) + centroids
        cost_train, acc_train, C_train = self.wmw_loss_train(
            P_train,
            self.sample2group_train,
            self.group_sizes_train,
            self.WMW_b,
        )

        reg = self.regularization()
        J = cost_train + self.lam * reg

        metrics = {
            "loss": J.detach().item(),
            "wmw_cost": cost_train.detach().item(),
            "reg": reg.detach().item(),
            "acc_train": acc_train.detach().item(),
        }

        # Validation metrics (using training centroids)
        if P_val is not None:
            cost_val, acc_val = self.wmw_loss_val(
                P_val,
                C_train,
                self.sample2group_val,
                self.WMW_b,
            )
            metrics["wmw_cost_val"] = cost_val.detach().item()
            metrics["acc_val"] = acc_val.detach().item()

        return J, metrics, Q, P_train, P_val, C_train


# ============================================================
# 3. Training loop helper
# ============================================================

def train_srw_torch_dense(
    edges,
    edge_features,
    P_init_train_csr,
    group_labels_train,
    rst_prob,
    lam=1e-3,
    norm_type="L1",
    WMW_b=2e-4,
    P_init_val_csr=None,
    group_labels_val=None,
    lr=0.1,
    max_epochs=100,
    early_stop=None,
    device=None,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert CSR -> dense torch matrices, with row-normalization
    P_init_train = csr_to_dense_torch(P_init_train_csr, device, renorm_rows=True)
    if P_init_val_csr is not None:
        P_init_val = csr_to_dense_torch(P_init_val_csr, device, renorm_rows=True)
    else:
        P_init_val = None

    # Build model
    model = SRW_solver(
        edges=edges,
        edge_features=edge_features,
        P_init_train=P_init_train,
        group_labels_train=group_labels_train,
        rst_prob=rst_prob,
        lam=lam,
        norm_type=norm_type,
        WMW_b=WMW_b,
        P_init_val=P_init_val,
        group_labels_val=group_labels_val,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam([model.w], lr=lr)
    history = []

    best_val_cost = None
    best_w = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        optimizer.zero_grad()

        J, metrics, Q, P_train, P_val, C_train = model()
        J.backward()
        optimizer.step()

        metrics["epoch"] = epoch
        metrics["time_sec"] = time.time() - t0
        history.append(metrics)

        # Progress print
        msg = (
            f"[Epoch {epoch:03d}] "
            f"loss={metrics['loss']:.4f} "
            f"wmw={metrics['wmw_cost']:.4f} "
            f"reg={metrics['reg']:.4f} "
            f"acc_train={metrics['acc_train']:.4f}"
        )
        if "wmw_cost_val" in metrics:
            msg += (
                f" | wmw_val={metrics['wmw_cost_val']:.4f} "
                f"acc_val={metrics['acc_val']:.4f}"
            )
        print(msg)

        # Early stopping based on validation WMW cost
        if P_init_val_csr is not None and early_stop is not None:
            val_cost = metrics["wmw_cost_val"]
            if (best_val_cost is None) or (val_cost < best_val_cost - 1e-6):
                best_val_cost = val_cost
                best_w = model.w.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop:
                    print(f"Early stopping at epoch {epoch}.")
                    if best_w is not None:
                        model.w.data.copy_(best_w)
                    break

    return model, history


# ============== Compatibility functions from SRW_v044.py ================

def logistic_edge_strength(features, w):
    """Return the edge strength (e by 1) calculated by a logistic function
    Inputs: edge features (e by w) and edge feature weights (vector w)"""
    if torch.is_tensor(w):
        w = w.detach().numpy()
    if hasattr(features, 'toarray'):
        features = features.toarray()
    return 1.0 / (1 + np.exp(-features.dot(w)))


def renorm(M):
    """Normalize a matrix by row sums, return a normalized matrix"""
    return csr_matrix(M / (M.sum(axis=1) + 1e-8))


def generate_Q(edges, nnodes, features, w):
    """Generate a transition matrix Q (n by n) without calculating gradients"""
    # Calculate edge strength
    edge_strength = logistic_edge_strength(features, w)
    
    # Convert edges to numpy array if it's a list
    if isinstance(edges, list):
        edges = np.array(edges)
    
    # M_strength (n by n) is a matrix containing edge strength
    # where M[i,j] = Strength[i,j];
    M_strength = csr_matrix((edge_strength, (edges[:, 0], edges[:, 1])),
                            shape=(nnodes, nnodes))
    Q = renorm(M_strength)
    return Q


def iterative_PPR(Q, P_init, rst_prob):
    """Takes P_init and a transition matrix to find the PageRank of nodes"""
    # Q and P_init are already normalized by row sums
    P = P_init.copy()
    rst_prob_P_init = rst_prob * P_init
    P_new = (1 - rst_prob) * np.dot(P, Q) + rst_prob_P_init
    
    max_iter = 1000
    tol = 1e-6
    for _ in range(max_iter):
        if np.allclose(P, P_new, atol=tol):
            break
        P = P_new
        P_new = (1 - rst_prob) * np.dot(P, Q) + rst_prob_P_init
    return P_new

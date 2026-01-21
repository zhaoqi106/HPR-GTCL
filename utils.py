import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_time_info(start_time: float, epoch: int, total_epochs: int, avg_epoch_time: float = None):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{now}] Epoch {epoch}/{total_epochs}")
    if avg_epoch_time is not None:
        remain = avg_epoch_time * (total_epochs - epoch)
        hrs = int(remain // 3600)
        mins = int((remain % 3600) // 60)
        secs = int(remain % 60)
        print(f"Estimated remaining time: {hrs}h {mins}m {secs}s (avg epoch {avg_epoch_time:.1f}s)")

def cosine_similarity_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return torch.matmul(x, y.t())

class InfoNCELoss(nn.Module):
    def __init__(self, init_temp: float = 0.07, hard_neg_k: int = 64, margin: float = 0.05):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp, dtype=torch.float32))
        self.hard_neg_k = int(hard_neg_k)
        self.margin = float(margin)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = z1.float()
        z2 = z2.float()
        N = z1.shape[0]
        if N == 0:
            return torch.tensor(0.0, device=z1.device, dtype=torch.float32)

        sim = cosine_similarity_matrix(z1, z2)
        device = sim.device

        diag_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim_masked = sim.masked_fill(diag_mask, float('-6e4'))

        pos = torch.diag(sim).unsqueeze(1)

        hard_k = min(self.hard_neg_k, sim_masked.shape[1] - 1)
        if hard_k <= 0:
            negs = sim_masked.clone()
            negs[negs < -1e4] = -1.0
        else:
            topk_vals, _ = torch.topk(sim_masked, k=hard_k, dim=1)
            negs = topk_vals

        logits = torch.cat([pos - self.margin, negs], dim=1)
        temp = torch.clamp(self.temperature, min=1e-4, max=10.0)
        logits = logits / temp
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
        return loss

def precision_recall_f1_at_k_numpy(y_true_np: np.ndarray, y_score_np: np.ndarray,
                                   K_list = range(1, 21)):
    results = {}
    N, H = y_true_np.shape
    for K in K_list:
        precisions = []
        recalls = []
        for i in range(N):
            true_idx = np.where(y_true_np[i] > 0.5)[0]
            if true_idx.size == 0:
                continue
            topk = np.argsort(-y_score_np[i])[:K]
            tp = len(np.intersect1d(topk, true_idx))
            prec = tp / K
            rec = tp / len(true_idx)
            precisions.append(prec)
            recalls.append(rec)
        if len(precisions) == 0:
            results[int(K)] = (0.0, 0.0, 0.0)
        else:
            mean_p = float(np.mean(precisions))
            mean_r = float(np.mean(recalls))
            if mean_p + mean_r > 0:
                mean_f1 = 2 * mean_p * mean_r / (mean_p + mean_r)
            else:
                mean_f1 = 0.0
            results[int(K)] = (mean_p, mean_r, mean_f1)
    return results

def node_drop_adj(adj_matrix: np.ndarray, drop_rate: float = 0.1, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    if drop_rate <= 0:
        return adj_matrix.copy()

    adj_aug = adj_matrix.copy()
    N = adj_matrix.shape[0]

    node_mask = np.random.rand(N) >= drop_rate
    nodes_to_drop = np.where(~node_mask)[0]

    if len(nodes_to_drop) > 0:
        adj_aug[nodes_to_drop, :] = 0.0
        adj_aug[:, nodes_to_drop] = 0.0

    deg = adj_aug.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    adj_aug = adj_aug / deg

    return adj_aug

def tsvd_adj(adj_matrix: np.ndarray, rank: int = 20, noise_std: float = 0.01, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    A = adj_matrix.astype(np.float32)
    N = A.shape[0]

    is_symmetric = np.allclose(A, A.T)

    try:
        U, S, VT = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        A_safe = A + np.random.normal(scale=1e-6, size=A.shape).astype(np.float32)
        U, S, VT = np.linalg.svd(A_safe, full_matrices=False)

    r = min(rank, len(S))

    if r > 0:
        A_recon = (U[:, :r] * S[:r]) @ VT[:r, :]
        A_recon = np.maximum(A_recon, 0)

        if is_symmetric:
            A_recon = (A_recon + A_recon.T) / 2

        if noise_std > 0:
            noise = np.random.normal(scale=noise_std, size=A_recon.shape).astype(np.float32)
            A_recon = A_recon + noise
            A_recon = np.maximum(A_recon, 0)
    else:
        A_recon = A.copy()

    deg = A_recon.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    A_recon = A_recon / deg

    return A_recon
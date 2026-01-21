import os
import numpy as np
import torch


def load_cities_edges(path: str):
    edges = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            edges.append((int(parts[0]), int(parts[1])))
    return edges


def build_dense_adj(num_nodes: int, edges, symmetric: bool = True, row_normalize: bool = True):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            A[u, v] = 1.0
            if symmetric:
                A[v, u] = 1.0
    if row_normalize:
        deg = A.sum(axis=1, keepdims=True)
        deg[deg == 0.0] = 1.0
        A = A / deg
    return torch.from_numpy(A)


def load_data(symptom_npy_path: str,
              herb_npy_path: str,
              symptom_cities_path: str,
              herb_cities_path: str,
              herb_offset_in_files: int = 390,
              device: str = 'cpu'):
    device = torch.device(device)
    symptom_feat = np.load(symptom_npy_path)
    herb_feat = np.load(herb_npy_path)

    sym_edges_raw = load_cities_edges(symptom_cities_path)
    herb_edges_raw = load_cities_edges(herb_cities_path)

    herb_edges_mapped = []
    for u, v in herb_edges_raw:
        herb_edges_mapped.append((u - herb_offset_in_files, v - herb_offset_in_files))

    sym_adj = build_dense_adj(num_nodes=symptom_feat.shape[0], edges=sym_edges_raw, symmetric=True, row_normalize=True)
    herb_adj = build_dense_adj(num_nodes=herb_feat.shape[0], edges=herb_edges_mapped, symmetric=True,
                               row_normalize=True)

    return {
        'symptom_feat': torch.from_numpy(symptom_feat).float().to(device),
        'herb_feat': torch.from_numpy(herb_feat).float().to(device),
        'sym_adj': sym_adj.to(device),
        'herb_adj': herb_adj.to(device),
        'sym_edges_raw': sym_edges_raw,
        'herb_edges_raw': herb_edges_raw
    }
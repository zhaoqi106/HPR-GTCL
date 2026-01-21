import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import csv

from dataloader import load_data
from model import HPRGTCL
from utils import set_seed, InfoNCELoss, precision_recall_f1_at_k_numpy, print_time_info, node_drop_adj, tsvd_adj

IN_DIR = "./Dataset2"
SYMPTOM_NPY = os.path.join(IN_DIR, "symptom.npy")
HERB_NPY = os.path.join(IN_DIR, "herb.npy")

SYMPTOM_CITIES = os.path.join(IN_DIR, "symptom.cities")
HERB_CITIES = os.path.join(IN_DIR, "herb.cities")

PD_CSV = os.path.join(IN_DIR, "Dataset2.csv")

OUT_DIR = r"./results"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "topk_results2.csv")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-5

EMBED_DIM = 256
TRANSFORMER_HEADS = 4
PROP_STEPS = 2
DROPOUT = 0.2
PROJ_DIM = 128

NODE_DROP_RATE = 0.15
TSVD_RANK = 40
TSVD_NOISE_STD = 0.01
HARD_NEG_K = 64
MARGIN = 0.05
INIT_TEMPERATURE = 0.07

HERB_OFFSET_IN_FILES = 560
TOPK = list(range(1, 21))

set_seed(SEED)

data_bundle = load_data(SYMPTOM_NPY, HERB_NPY,
                        SYMPTOM_CITIES, HERB_CITIES,
                        herb_offset_in_files=HERB_OFFSET_IN_FILES,
                        device=DEVICE)

symptom_feat = data_bundle['symptom_feat']
herb_feat = data_bundle['herb_feat']
sym_adj = data_bundle['sym_adj']
herb_adj = data_bundle['herb_adj']

df = pd.read_csv(PD_CSV)
sym_col = df.columns[0]
herb_col = df.columns[1]

def parse_cell_to_list(cell):
    if isinstance(cell, str) and cell.strip() != "":
        return [int(x) for x in cell.strip().split()]
    return []

symptom_lists = []
herb_lists_adjusted = []
for _, row in df.iterrows():
    s = parse_cell_to_list(row[sym_col])
    h_raw = parse_cell_to_list(row[herb_col])
    h = [int(x) - HERB_OFFSET_IN_FILES for x in h_raw]
    symptom_lists.append(s)
    herb_lists_adjusted.append(h)

H = herb_feat.size(0)
Y = np.zeros((len(herb_lists_adjusted), H), dtype=np.float32)
for i, hlist in enumerate(herb_lists_adjusted):
    for h in hlist:
        if 0 <= h < H:
            Y[i, h] = 1.0

idx = np.arange(len(df))
train_idx, tmp_idx = train_test_split(idx, train_size=0.8, random_state=SEED, shuffle=True)
val_idx, test_idx = train_test_split(tmp_idx, train_size=0.5, random_state=SEED, shuffle=True)

X_train_sym = [symptom_lists[i] for i in train_idx]
y_train = Y[train_idx]
X_val_sym = [symptom_lists[i] for i in val_idx]
y_val = Y[val_idx]
X_test_sym = [symptom_lists[i] for i in test_idx]
y_test = Y[test_idx]

print(f"Dataset sizes -> train: {len(X_train_sym)} val: {len(X_val_sym)} test: {len(X_test_sym)}")

model = HPRGTCL(
    symptom_feat_dim=symptom_feat.size(1),
    herb_feat_dim=herb_feat.size(1),
    embed_dim=EMBED_DIM,
    transformer_heads=TRANSFORMER_HEADS,
    prop_steps=PROP_STEPS,
    dropout=DROPOUT,
    proj_dim=PROJ_DIM
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=False)
bce_loss = nn.BCEWithLogitsLoss()
info_nce = InfoNCELoss(init_temp=INIT_TEMPERATURE, hard_neg_k=HARD_NEG_K, margin=MARGIN).to(DEVICE)
scaler = GradScaler(enabled=torch.cuda.is_available())

start_time = time.time()
epoch_times = []
best_val = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    perm = np.random.permutation(len(X_train_sym))
    total_loss = 0.0

    for bstart in range(0, len(perm), BATCH_SIZE):
        bidx = perm[bstart: bstart + BATCH_SIZE]
        batch_sym_lists = [X_train_sym[i] for i in bidx]
        batch_y = torch.tensor(y_train[bidx], dtype=torch.float32, device=DEVICE)

        sym_adj_np = sym_adj.cpu().numpy()
        herb_adj_np = herb_adj.cpu().numpy()

        sym_adj_view1_np = node_drop_adj(sym_adj_np, drop_rate=NODE_DROP_RATE, seed=epoch)
        herb_adj_view1_np = node_drop_adj(herb_adj_np, drop_rate=NODE_DROP_RATE, seed=epoch + 1)

        sym_adj_view2_np = tsvd_adj(
            sym_adj_np,
            rank=min(TSVD_RANK, sym_adj_np.shape[0]),
            noise_std=TSVD_NOISE_STD,
            seed=epoch + 2
        )
        herb_adj_view2_np = tsvd_adj(
            herb_adj_np,
            rank=min(TSVD_RANK, herb_adj_np.shape[0]),
            noise_std=TSVD_NOISE_STD,
            seed=epoch + 3
        )

        sym_adj_view1 = torch.from_numpy(sym_adj_view1_np).float().to(DEVICE)
        sym_adj_view2 = torch.from_numpy(sym_adj_view2_np).float().to(DEVICE)
        herb_adj_view1 = torch.from_numpy(herb_adj_view1_np).float().to(DEVICE)
        herb_adj_view2 = torch.from_numpy(herb_adj_view2_np).float().to(DEVICE)

        optimizer.zero_grad()
        with autocast(enabled=torch.cuda.is_available()):
            logits = model(
                batch_sym_lists,
                symptom_feat, herb_feat,
                sym_adj=sym_adj, herb_adj=herb_adj
            )
            loss_pred = bce_loss(logits, batch_y)

            with torch.no_grad():
                sym_emb_v1 = model.sym_input(symptom_feat)
                for transformer in model.sym_transformers:
                    sym_emb_v1 = transformer(sym_emb_v1, sym_adj_view1)
                for p in model.sym_prop:
                    sym_emb_v1 = p(sym_emb_v1, sym_adj_view1)

                sym_emb_v2 = model.sym_input(symptom_feat)
                for transformer in model.sym_transformers:
                    sym_emb_v2 = transformer(sym_emb_v2, sym_adj_view2)
                for p in model.sym_prop:
                    sym_emb_v2 = p(sym_emb_v2, sym_adj_view2)

                herb_emb_v1 = model.herb_input(herb_feat)
                for transformer in model.herb_transformers:
                    herb_emb_v1 = transformer(herb_emb_v1, herb_adj_view1)
                for p in model.herb_prop:
                    herb_emb_v1 = p(herb_emb_v1, herb_adj_view1)

                herb_emb_v2 = model.herb_input(herb_feat)
                for transformer in model.herb_transformers:
                    herb_emb_v2 = transformer(herb_emb_v2, herb_adj_view2)
                for p in model.herb_prop:
                    herb_emb_v2 = p(herb_emb_v2, herb_adj_view2)

            sym_proj_v1 = model.sym_projector(sym_emb_v1)
            sym_proj_v2 = model.sym_projector(sym_emb_v2)
            herb_proj_v1 = model.herb_projector(herb_emb_v1)
            herb_proj_v2 = model.herb_projector(herb_emb_v2)

            loss_sym_con = info_nce(sym_proj_v1, sym_proj_v2)
            loss_herb_con = info_nce(herb_proj_v1, herb_proj_v2)

            loss = loss_pred + loss_sym_con + loss_herb_con

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * len(bidx)

    avg_train_loss = total_loss / len(X_train_sym)
    epoch_times.append(time.time() - t0)

    model.eval()
    with torch.no_grad():
        val_scores_parts = []
        for bstart in range(0, len(X_val_sym), BATCH_SIZE):
            batch_lists = X_val_sym[bstart: bstart + BATCH_SIZE]
            logits_val = model(
                batch_lists,
                symptom_feat, herb_feat,
                sym_adj=sym_adj, herb_adj=herb_adj
            )
            val_scores_parts.append(logits_val.detach().cpu().numpy())
        val_scores = np.vstack(val_scores_parts)
    val_metrics = precision_recall_f1_at_k_numpy(y_val, val_scores, K_list=[1, 5, 10])
    val_f1_10 = val_metrics[10][2]
    scheduler.step(val_f1_10)

    avg_epoch_time = float(np.mean(epoch_times[-10:])) if len(epoch_times) > 0 else None
    print_time_info(start_time, epoch, EPOCHS, avg_epoch_time=avg_epoch_time)
    print(f"Epoch {epoch} TrainLoss={avg_train_loss:.4f} ValF1@10={val_f1_10:.4f}")

    if val_f1_10 > best_val + 1e-8:
        best_val = val_f1_10
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(best_state, os.path.join(OUT_DIR, "best_model_adj_tsvd_node_drop.pth"))
        print(f"Saved best model at epoch {epoch} (Val F1@10={best_val:.4f})")

if best_state is not None:
    model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_scores_parts = []
    for bstart in range(0, len(X_test_sym), BATCH_SIZE):
        batch_lists = X_test_sym[bstart: bstart + BATCH_SIZE]
        logits_test = model(
            batch_lists,
            symptom_feat, herb_feat,
            sym_adj=sym_adj, herb_adj=herb_adj
        )
        test_scores_parts.append(logits_test.detach().cpu().numpy())
    test_scores = np.vstack(test_scores_parts)

test_metrics = precision_recall_f1_at_k_numpy(y_test, test_scores, K_list=TOPK)

with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["K", "Precision", "Recall", "F1"])
    for K in TOPK:
        p, r, f = test_metrics[K]
        writer.writerow([K, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"])
print(f"Saved Top@K results to {OUT_CSV}")

print("Finished.")
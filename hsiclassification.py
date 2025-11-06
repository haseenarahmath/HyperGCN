from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.io as scio
import sklearn.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

try:
    # Optional, provided in the original repo
    from layers import PairNorm  # type: ignore
except Exception:  # graceful fallback
    class PairNorm(nn.Module):  # minimal no‑op drop‑in replacement
        def __init__(self, mode: str = "None", scale: float = 1.0):
            super().__init__()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

from torch_geometric.nn import GCNConv  # used by GCN/GCN4


# -----------------------------
# Utility helpers
# -----------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    dataset: str = "in"  # {"in", "pav"}
    model: str = "DeepGCN"  # {"GCN","GCN4","MultiGCN","GCNII","DeepGCN"}
    hidden_dim: int = 128
    n_layers: int = 5
    dropout: float = 0.06
    residual: int = 1
    norm_mode: str = "PN-SI"
    norm_scale: float = 1.0
    lr: float = 5e-3
    weight_decay: float = 5e-4
    epochs: int = 1000
    patience: int = 50
    seed: int = 42
    out_dir: Path = Path("results")
    checkpoint_name: str = "checkpoint-best-acc.pt"


# -----------------------------
# Graph layers used by DeepGCN/MultiGCN/GCNII
# -----------------------------
class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(1, out_features)) if bias else None
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x @ self.weight
        out = torch.spmm(adj, h)
        return out + self.bias if self.bias is not None else out


class GraphConvolution(nn.Module):
    """GCNII building block."""
    def __init__(self, in_features: int, out_features: int, residual: bool = False, variant: bool = False):
        super().__init__()
        self.variant = variant
        self.residual = residual
        self.in_features = 2 * in_features if variant else in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.out_features)
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, h0: torch.Tensor, lamda: float, alpha: float, layer_idx: int) -> torch.Tensor:
        theta = math.log(lamda / layer_idx + 1)
        hi = torch.spmm(adj, x)
        if self.variant:
            support = torch.cat([hi, h0], dim=1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        out = theta * (support @ self.weight) + (1 - theta) * r
        if self.residual:
            out = out + x
        return out


# -----------------------------
# Models
# -----------------------------
class DeepGCN(nn.Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float, nlayer: int = 2, residual: int = 0,
                 norm_mode: str = "None", norm_scale: float = 1.0):
        super().__init__()
        assert nlayer >= 1
        self.hidden_layers = nn.ModuleList([GraphConv(nhid, nhid) for _ in range(nlayer)])
        self.fc_in = nn.Linear(nfeat, nhid)
        self.fc_out = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_tail = nn.Dropout(p=0.4)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.skip = residual
        self.bn_in = nn.BatchNorm1d(nfeat)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x_old = torch.zeros_like(x)
        x = self.bn_in(x)
        x = self.fc_in(x)
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip > 0 and i % self.skip == 0:
                x = x + x_old
                x_old = x
        x = self.dropout_tail(x)
        x = self.fc_out(x)
        return x  # logits


class MultiGCN(nn.Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float, nlayer: int = 2):
        super().__init__()
        assert nlayer >= 1
        self.hidden_layers = nn.ModuleList([GraphConv(nhid, nhid) for _ in range(nlayer)])
        self.fc_in = nn.Linear(nfeat, nhid)
        self.fc_out = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_tail = nn.Dropout(p=0.4)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        for layer in self.hidden_layers:
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.relu(x)
        x = self.dropout_tail(x)
        x = self.fc_out(x)
        return x


class GCNII(nn.Module):
    def __init__(self, nfeat: int, nhidden: int, nclass: int, dropout: float, nlayers: int, lamda: float, alpha: float, variant: bool):
        super().__init__()
        self.convs = nn.ModuleList([GraphConvolution(nhidden, nhidden, variant=variant) for _ in range(nlayers)])
        self.fc_in = nn.Linear(nfeat, nhidden)
        self.fc_out = nn.Linear(nhidden, nclass)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.alpha = alpha
        self.lamda = lamda
        self.dropout_tail = nn.Dropout(p=0.4)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h0 = self.fc_in(x)
        h = h0
        for i, conv in enumerate(self.convs, start=1):
            h = self.dropout(conv(h, adj, h0, self.lamda, self.alpha, i))
            h = self.relu(h)
        h = self.dropout_tail(h)
        h = self.fc_out(h)
        return h


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float, n_graph_layers: int = 2):
        super().__init__()
        self.drop1 = nn.Dropout(dropout_p)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout_p)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.drop1(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.conv2(x, edge_index)
        return x


class GCN4(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float, n_graph_layers: int = 2):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(input_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn_h = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.bn_in(x)
        x = self.conv1(x, edge_index)
        x = self.bn_h(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        return x


# -----------------------------
# Data loading & preprocessing
# -----------------------------

def load_data(ds: str) -> Tuple[Dict, Dict, Dict]:
    if ds == "in":
        ALL_X = scio.loadmat("data/ALL_X.mat")
        ALL_Y = scio.loadmat("data/ALL_Y.mat")
        ALL_L = scio.loadmat("data/ALL_L.mat")
    elif ds == "pav":
        ALL_X = scio.loadmat("data/New_Pav_ALL_X.mat")
        ALL_Y = scio.loadmat("data/New_Pav_ALL_Y.mat")
        ALL_L = scio.loadmat("data/New_Pav_ALL_L.mat")
    else:
        raise ValueError(f"Unknown dataset: {ds}")
    return ALL_X, ALL_Y, ALL_L


def sample_mask(idx: np.ndarray, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    mask[idx] = True
    return mask


def get_train_test_sizes(ds: str) -> Tuple[int, int]:
    if ds == "in":
        return 695, 10366
    if ds == "pav":
        return 2774, 42776
    raise ValueError(f"Unknown dataset: {ds}")


def preprocess(ALL_X: Dict, ALL_Y: Dict, ALL_L: Dict, num_classes: int | None, train_c: int, test_c: int, device: torch.device):
    L = torch.from_numpy(ALL_L["ALL_L"].todense()).float().to(device)
    X = torch.from_numpy(ALL_X["ALL_X"]).float().to(device)
    Y_int = ALL_Y["ALL_Y"].astype(int).reshape(-1)  # 1‑based labels

    # derive num_classes if not provided
    if num_classes is None:
        num_classes = int(Y_int.max())

    # one‑hot
    Y_oh = np.eye(num_classes)[(Y_int - 1)]  # shift to 0‑based
    Y = torch.from_numpy(Y_oh).float().to(device)

    # masks
    tr_mask = torch.from_numpy(sample_mask(np.arange(0, train_c), Y.shape[0])).to(device)
    te_mask = torch.from_numpy(sample_mask(np.arange(train_c, test_c), Y.shape[0])).to(device)
    # keep a shuffle for legacy parity
    tr_mask, te_mask = sklearn.utils.shuffle(tr_mask, te_mask)

    n_x = X.shape[1]
    n_y = Y.shape[1]
    return n_x, n_y, tr_mask, te_mask, X, Y, L


# -----------------------------
# Metrics
# -----------------------------

def accuracy_from_logits(logits: torch.Tensor, labels_oh: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    labels = labels_oh.argmax(dim=1)
    return (preds == labels).float().mean().item()


def per_class_accuracy(logits: torch.Tensor, labels_oh: torch.Tensor) -> np.ndarray:
    preds = logits.argmax(dim=1).cpu().numpy()
    labels = labels_oh.argmax(dim=1).cpu().numpy()
    cm = np.zeros((labels_oh.shape[1], labels_oh.shape[1]), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    accs = []
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        accs.append((cm[i, i] / denom) if denom > 0 else 0.0)
    return np.array(accs)


def sklearn_prf_kappa(logits: torch.Tensor, labels_oh: torch.Tensor) -> Tuple[float, float, float, float]:
    preds = logits.argmax(dim=1).cpu().numpy()
    labels = labels_oh.argmax(dim=1).cpu().numpy()
    P = precision_score(labels, preds, average="weighted", zero_division=0)
    R = recall_score(labels, preds, average="weighted", zero_division=0)
    F = f1_score(labels, preds, average="weighted", zero_division=0)
    K = cohen_kappa_score(labels, preds)
    return P, R, F, K


# -----------------------------
# Train / Val steps
# -----------------------------
@torch.no_grad()
def val_step(model: nn.Module, X: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor, Y: torch.Tensor, loss_fn: nn.Module) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    logits = model(X, adj)
    loss = loss_fn(logits[mask], Y[mask]).item()
    acc = accuracy_from_logits(logits[mask], Y[mask])
    return loss, acc, logits[mask]


def train_step(model: nn.Module, X: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor, Y: torch.Tensor, loss_fn: nn.Module, opt: torch.optim.Optimizer) -> Tuple[float, float]:
    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model(X, adj)
    loss = loss_fn(logits[mask], Y[mask])
    acc = accuracy_from_logits(logits[mask], Y[mask])
    loss.backward()
    opt.step()
    return loss.item(), acc


# -----------------------------
# Model factory (adj‑based models only, matching the original training path)
# -----------------------------

def build_model(name: str, n_x: int, hidden: int, n_y: int, cfg: TrainConfig) -> nn.Module:
    if name == "DeepGCN":
        return DeepGCN(n_x, hidden, n_y, cfg.dropout, nlayer=cfg.n_layers, residual=cfg.residual, norm_mode=cfg.norm_mode, norm_scale=cfg.norm_scale)
    if name == "MultiGCN":
        return MultiGCN(n_x, hidden, n_y, cfg.dropout, nlayer=cfg.n_layers)
    if name == "GCNII":
        return GCNII(nfeat=n_x, nhidden=hidden, nclass=n_y, dropout=cfg.dropout, nlayers=cfg.n_layers, lamda=0.5, alpha=0.1, variant=True)
    # Edge‑index models (GCN/GCN4) are not wired here since data comes as dense adjacency.
    raise ValueError(f"Model '{name}' expects dense adj training path. Choose one of: DeepGCN, MultiGCN, GCNII.")


# -----------------------------
# Runner
# -----------------------------

def run(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg.out_dir)

    ALL_X, ALL_Y, ALL_L = load_data(cfg.dataset)
    train_c, test_c = get_train_test_sizes(cfg.dataset)

    n_x, n_y, tr_mask, te_mask, X, Y, L = preprocess(ALL_X, ALL_Y, ALL_L, num_classes=None, train_c=train_c, test_c=test_c, device=device)
    model = build_model(cfg.model, n_x, cfg.hidden_dim, n_y, cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_logits_val = None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_step(model, X, tr_mask, L, Y, loss_fn, opt)
        val_loss, val_acc, logits_val = val_step(model, X, te_mask, L, Y, loss_fn)

        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            best_logits_val = logits_val.detach().cpu()
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch,
                "best_val_acc": best_acc,
            }, cfg.out_dir / cfg.checkpoint_name)
        else:
            epochs_no_improve += 1

        print(f"[Epoch {epoch:04d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} {'*' if improved else ''}")

        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {cfg.patience} epochs). Best epoch: {best_epoch}, best val acc: {best_acc:.4f}")
            break

    # Final metrics on validation mask (legacy script used this as 'test')
    assert best_logits_val is not None, "Training finished without validation forward pass."
    P, R, F, K = sklearn_prf_kappa(best_logits_val, Y[te_mask])
    pc_acc = per_class_accuracy(best_logits_val, Y[te_mask]).tolist()

    summary = {
        "dataset": cfg.dataset,
        "model": cfg.model,
        "hidden_dim": cfg.hidden_dim,
        "n_layers": cfg.n_layers,
        "dropout": cfg.dropout,
        "norm_mode": cfg.norm_mode,
        "best_val_acc": round(best_acc, 6),
        "best_epoch": best_epoch,
        "precision_w": round(P, 6),
        "recall_w": round(R, 6),
        "f1_w": round(F, 6),
        "cohen_kappa": round(K, 6),
        "per_class_acc": pc_acc,
    }

    with open(cfg.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", cfg.out_dir / cfg.checkpoint_name, "and", cfg.out_dir / "summary.json")


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HyperGCN HSI Classification — Clean Training Script")
    p.add_argument("--dataset", choices=["in", "pav"], default="in")
    p.add_argument("--model", choices=["DeepGCN", "MultiGCN", "GCNII"], default="DeepGCN")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.06)
    p.add_argument("--residual", type=int, default=1)
    p.add_argument("--norm_mode", type=str, default="PN-SI")
    p.add_argument("--norm_scale", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    p.add_argument("--checkpoint_name", type=str, default="checkpoint-best-acc.pt")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TrainConfig(**vars(args))
    run(cfg)


if __name__ == "__main__":
    main()

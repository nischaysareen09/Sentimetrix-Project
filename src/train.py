"""
train.py
========
Real training pipeline for SentimetrixTCN. Rewritten from the original with
several correctness fixes:

1. TIME-BASED SPLIT (not random). The original code fit the scaler on ALL
   data (train+val combined) then split — that's train/test leakage: the
   scaler "saw" statistics from the future. Here the scaler is fit ONLY on
   the training slice, and train/val/test are split chronologically per
   asset (earliest 70% train, next 15% val, final 15% test) so nothing from
   the future ever leaks into training, matching how the model would
   actually be used in production (predict forward from the past only).

2. Early stopping + LR scheduling on a held-out validation set, with the
   TEST set touched exactly once, at the very end, for the reported metric.

3. Honest metrics.json: written from the untouched test set, with a full
   classification report + confusion matrix, not a single made-up number.

This script trains on the synthetic multi-regime market simulator
(synthetic_market.py) because this environment has no network access to
real market data providers. See train_real_data.py for the equivalent
pipeline against live Yahoo Finance history — run that locally to get
production-grade weights before a real demo.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from synthetic_market import generate_synthetic_universe
from market_data import compute_indicators
from model_engine import SentimetrixTCN
from rag_engine import EXPERT_RULES, embedder

SEQ_LEN = 20
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8
LEARNING_RATE = 1e-3
N_ASSETS = 60
N_DAYS = 2000

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SAVE_PATH = os.path.join(MODELS_DIR, "alpha_weights.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(PROJECT_ROOT, "metrics.json")


class AlphaDataset(Dataset):
    def __init__(self, x_data, y_data, rules_embeddings):
        self.x_data = x_data
        self.y_data = y_data
        self.rules_embeddings = rules_embeddings

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_price = torch.tensor(self.x_data[idx], dtype=torch.float32).transpose(0, 1)
        x_rag = torch.tensor(self.rules_embeddings, dtype=torch.float32)
        label = torch.tensor(self.y_data[idx], dtype=torch.long)
        return x_price, x_rag, label


def build_asset_sequences(df, seq_len):
    """Given one asset's indicator+label dataframe (already time-sorted),
    returns (X, y) sliding-window sequences."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ("target", "returns"):
        if col in numeric_cols:
            numeric_cols.remove(col)

    values = df[numeric_cols].values
    targets = df["target"].values

    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i + seq_len])
        y.append(targets[i + seq_len - 1])
    return np.array(X), np.array(y), numeric_cols


def time_split(n, train_frac=0.70, val_frac=0.15):
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return train_end, val_end


def prepare_data():
    print(f"Generating synthetic universe: {N_ASSETS} assets x {N_DAYS} days...")
    universe = generate_synthetic_universe(n_assets=N_ASSETS, n_days=N_DAYS, seed=42)

    rule_embeddings = embedder.encode(EXPERT_RULES)

    per_asset = {}  # ticker -> (train_slice, val_slice, test_slice) with 'returns' but no 'target' yet

    for ticker, raw_df in universe.items():
        df = compute_indicators(raw_df)
        df["returns"] = df["close"].pct_change().shift(-1)  # next-day forward return
        df = df.dropna()
        if len(df) < (SEQ_LEN + 50):
            continue

        # Chronological split PER ASSET on the raw indicator frame, before
        # sequencing, so no window straddles a split boundary in a way that
        # leaks test-period data into a training window.
        n = len(df)
        train_end, val_end = time_split(n)

        train_slice = df.iloc[:train_end]
        val_slice = df.iloc[max(0, train_end - SEQ_LEN):val_end]
        test_slice = df.iloc[max(0, val_end - SEQ_LEN):]
        per_asset[ticker] = (train_slice, val_slice, test_slice)

    # --- Quantile-based labeling ---------------------------------------
    # The original fixed +-2.5% thresholds made "Hold" ~97% of samples
    # (a 2.5% single-day move is a rare tail event at ~1.5% daily vol),
    # which starves the model of Sell/Buy examples and makes training
    # collapse. Instead: compute the 33rd/67th percentile of forward
    # returns from the TRAINING portion only (never val/test — that would
    # be leakage), and use those as the Sell/Hold/Buy cutoffs. This gives
    # a naturally balanced ~33/33/33 class split, which is standard
    # practice in quant return-classification research.
    all_train_returns = np.concatenate([s[0]["returns"].values for s in per_asset.values()])
    down_thresh, up_thresh = np.quantile(all_train_returns, [1 / 3, 2 / 3])
    print(f"Quantile label thresholds (from train only): down={down_thresh:.4f}, up={up_thresh:.4f}")

    def label(df):
        df = df.copy()
        df["target"] = df["returns"].apply(lambda r: 2 if r > up_thresh else (0 if r < down_thresh else 1))
        return df

    train_X, train_y = [], []
    val_X, val_y = [], []
    test_X, test_y = [], []
    feature_cols = None

    for ticker, (train_slice, val_slice, test_slice) in per_asset.items():
        train_slice, val_slice, test_slice = label(train_slice), label(val_slice), label(test_slice)

        tX, ty, cols = build_asset_sequences(train_slice, SEQ_LEN)
        vX, vy, _ = build_asset_sequences(val_slice, SEQ_LEN)
        teX, tey, _ = build_asset_sequences(test_slice, SEQ_LEN)

        if feature_cols is None:
            feature_cols = cols

        if len(tX): train_X.append(tX); train_y.append(ty)
        if len(vX): val_X.append(vX); val_y.append(vy)
        if len(teX): test_X.append(teX); test_y.append(tey)

    train_X = np.concatenate(train_X); train_y = np.concatenate(train_y)
    val_X = np.concatenate(val_X); val_y = np.concatenate(val_y)
    test_X = np.concatenate(test_X); test_y = np.concatenate(test_y)

    print(f"Sequences -> train: {len(train_X)}, val: {len(val_X)}, test: {len(test_X)}")
    print(f"Feature dim: {train_X.shape[2]} ({len(feature_cols)} indicator columns)")

    # Fit scaler on TRAIN ONLY (the original bug: scaler was fit on
    # train+val combined before splitting, leaking val-period statistics
    # into the "unseen" data the model was later evaluated on).
    n_feat = train_X.shape[2]
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, n_feat))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler fit on train-only data, saved to {SCALER_PATH}")

    def scale(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, shape[2])).reshape(shape)

    train_X, val_X, test_X = scale(train_X), scale(val_X), scale(test_X)

    return (train_X, train_y), (val_X, val_y), (test_X, test_y), rule_embeddings, n_feat


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x_p, x_r, y in loader:
            x_p, x_r, y = x_p.to(device), x_r.to(device), y.to(device)
            logits, _ = model(x_p, x_r)
            total_loss += criterion(logits, y).item()
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return y_true, y_pred, total_loss / max(len(loader), 1)


def train_model():
    (train_X, train_y), (val_X, val_y), (test_X, test_y), rule_embeddings, n_feat = prepare_data()

    train_loader = DataLoader(AlphaDataset(train_X, train_y, rule_embeddings), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AlphaDataset(val_X, val_y, rule_embeddings), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(AlphaDataset(test_X, test_y, rule_embeddings), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | feature_dim={n_feat}")

    class_counts = np.bincount(train_y.astype(int), minlength=3)
    class_weights = len(train_y) / (3 * np.maximum(class_counts, 1))
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Train class distribution (Sell/Hold/Buy): {class_counts.tolist()} -> weights {np.round(class_weights, 3).tolist()}")

    model = SentimetrixTCN(input_size=n_feat).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_p, x_r, y in train_loader:
            x_p, x_r, y = x_p.to(device), x_r.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x_p, x_r)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        val_true, val_pred, val_loss = evaluate(model, val_loader, device)
        val_acc = 100 * np.mean(np.array(val_true) == np.array(val_pred))
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] train_loss={total_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.2f}% val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no val improvement for {PATIENCE} epochs)")
                break

    # Restore best checkpoint (lowest val loss), then evaluate ONCE on the
    # untouched test set for the final reported metric.
    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Best model (val_loss={best_val_loss:.4f}) saved to {SAVE_PATH}")

    test_true, test_pred, test_loss = evaluate(model, test_loader, device)
    report = classification_report(
        test_true, test_pred, labels=[0, 1, 2],
        target_names=["Sell", "Hold", "Buy"],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(test_true, test_pred, labels=[0, 1, 2]).tolist()

    print("\n=== HELD-OUT TEST SET RESULTS (touched once, never used for training/model-selection) ===")
    print(classification_report(test_true, test_pred, labels=[0, 1, 2], target_names=["Sell", "Hold", "Buy"], zero_division=0))
    print("Confusion matrix (rows=true, cols=pred) [Sell, Hold, Buy]:")
    for row in cm:
        print(" ", row)

    metrics = {
        "data_source": "synthetic_regime_switching_simulator (see src/synthetic_market.py)",
        "note": "Trained on simulated data due to no live market data access in the training environment. Re-run train_real_data.py locally against live Yahoo Finance history for production metrics.",
        "test_accuracy": report["accuracy"],
        "test_macro_precision": report["macro avg"]["precision"],
        "test_macro_recall": report["macro avg"]["recall"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            "Sell": report["Sell"],
            "Hold": report["Hold"],
            "Buy": report["Buy"],
        },
        "confusion_matrix": {"labels": ["Sell", "Hold", "Buy"], "matrix": cm},
        "n_train_sequences": int(len(train_X)),
        "n_val_sequences": int(len(val_X)),
        "n_test_sequences": int(len(test_X)),
        "n_assets": N_ASSETS,
        "seq_len": SEQ_LEN,
        "feature_dim": int(n_feat),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {METRICS_PATH}")
    print(f"Final test accuracy: {report['accuracy']*100:.2f}% | macro F1: {report['macro avg']['f1-score']:.3f}")


if __name__ == "__main__":
    train_model()

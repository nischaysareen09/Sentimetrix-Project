"""
model_engine.py
================
SentimetrixTCN v2 — a real Temporal Convolutional Network (Bai et al., 2018
style: causal dilated convolutions + residual blocks + weight norm), fused
with retrieved expert-rule embeddings via scaled dot-product attention.

What changed vs v1 and why:
- v1's "TCN" was two plain Conv1d layers with no residual connections and
  no causal masking — not actually a TCN, just a small conv stack. Real TCN
  power comes from stacking *causal* dilated residual blocks so the
  receptive field grows exponentially with depth while still respecting
  time order (no peeking at future timesteps).
- v1 had zero regularization (no dropout, no norm layers anywhere), which
  on a ~20-timestep window with only a few hundred training sequences per
  ticker is a near-guaranteed overfit. Added dropout + BatchNorm/LayerNorm
  throughout.
- v1's attention scaled scores by a hardcoded `/ 11.3` (a stand-in for
  sqrt(128)). Fixed to use the actual sqrt(d_k) of whatever dimension is
  configured, and added attention dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # older torch versions
    from torch.nn.utils import weight_norm
import numpy as np


# ---------------------------------------------------------------------------
# Causal TCN building blocks (Bai, Kolter, Koltun 2018 — "An Empirical
# Evaluation of Generic Convolutional and Recurrent Networks for Sequence
# Modeling")
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    """Removes the extra right-padding so a 'same'-padded causal conv only
    ever looks at the past, never the future."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """One residual block: two causal dilated convs + a residual (skip)
    connection, matching the standard TCN residual unit."""

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu_out = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.drop1(self.relu1(self.norm1(self.chomp1(self.conv1(x)))))
        out = self.drop2(self.relu2(self.norm2(self.chomp2(self.conv2(out)))))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + residual)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially growing dilation
    (1, 2, 4, 8, ...), giving an exponentially growing receptive field."""

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# Cross-modal fusion: price features attend over retrieved expert-rule
# embeddings (scaled dot-product attention, correctly scaled this time).
# ---------------------------------------------------------------------------

class AlphaAttention(nn.Module):
    def __init__(self, tcn_dim=128, rag_dim=384, attn_dim=128, dropout=0.1):
        super().__init__()
        self.attn_dim = attn_dim
        self.query = nn.Linear(tcn_dim, attn_dim)
        self.key = nn.Linear(rag_dim, attn_dim)
        self.value = nn.Linear(rag_dim, attn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tcn_feat, rag_feat):
        Q = self.query(tcn_feat)
        K = self.key(rag_feat)
        V = self.value(rag_feat)

        # Correctly scaled dot-product attention: divide by sqrt(d_k), not
        # a hardcoded magic number. Keeps gradients well-behaved regardless
        # of attn_dim if that ever changes.
        scale = self.attn_dim ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, V)
        return context, weights


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SentimetrixTCN(nn.Module):
    def __init__(self, input_size=19, tcn_channels=(64, 96, 128, 128),
                 rag_dim=384, dropout=0.25, num_classes=3):
        super().__init__()

        self.tcn = TemporalConvNet(input_size, list(tcn_channels), kernel_size=3, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

        tcn_out_dim = tcn_channels[-1]
        self.fusion = AlphaAttention(tcn_dim=tcn_out_dim, rag_dim=rag_dim, attn_dim=128, dropout=dropout * 0.5)

        combined_dim = tcn_out_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x_price, x_rag):
        """
        x_price: (batch, input_size, seq_len)  — price/indicator sequence
        x_rag:   (batch, num_rules, rag_dim)    — retrieved rule embeddings
        """
        feat = self.tcn(x_price)                 # (batch, channels, seq_len)
        p_feat = self.pool(feat).transpose(1, 2)  # (batch, 1, channels)

        r_context, weights = self.fusion(p_feat, x_rag)  # (batch, 1, 128)
        combined = torch.cat([p_feat, r_context], dim=-1).squeeze(1)  # (batch, channels+128)

        logits = self.classifier(combined)
        return logits, weights


# ---------------------------------------------------------------------------
# Inference helper (used by the API layer)
# ---------------------------------------------------------------------------

def predict_signal(model, df, rules, embedder, scaler=None, seq_len=20):
    """Runs a single forward pass over the most recent `seq_len` rows of
    `df` plus the retrieved `rules`, returning (predicted_class, confidence_%,
    attention_weights)."""
    numeric_df = df.select_dtypes(include=[np.number])
    for col in ("target", "returns"):
        if col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[col])

    recent_data = numeric_df.values[-seq_len:]
    if len(recent_data) < seq_len:
        return 1, 50.0, np.zeros(max(len(rules), 1))

    recent_data = np.nan_to_num(recent_data)

    n_features = model.tcn.network[0].conv1.in_channels if hasattr(model, "tcn") else 19
    if recent_data.shape[1] < n_features:
        padding = np.zeros((recent_data.shape[0], n_features - recent_data.shape[1]))
        recent_data = np.hstack((recent_data, padding))
    elif recent_data.shape[1] > n_features:
        recent_data = recent_data[:, :n_features]

    if scaler is not None:
        try:
            recent_data = scaler.transform(recent_data)
        except Exception:
            recent_data = np.nan_to_num(recent_data)

    x_price = torch.tensor(recent_data.T, dtype=torch.float32).unsqueeze(0)

    try:
        rule_embeddings = embedder.encode(rules)
    except Exception:
        rule_embeddings = np.zeros((max(len(rules), 1), 384))

    x_rag = torch.tensor(rule_embeddings, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(x_price, x_rag)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    return predicted_class.item(), confidence.item() * 100, attn_weights.squeeze().numpy()

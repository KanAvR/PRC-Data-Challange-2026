"""Neural regressor for the normal regime, as a fifth `mu` member.

The four tree members' residuals correlate 0.94-0.97, and every additional tree
variant measured so far (XGBoost, `lgb_nocat`, both residual anchors) has been
better in isolation and a wash inside the average. A network makes genuinely
different errors: it fits a smooth function of the numeric gaps instead of a step
function, and it learns a dense embedding per stand/runway/airline rather than
splitting on them.

CPU only, no ROCm needed: 1.4M rows and ~100 features train in a couple of
minutes per epoch on 16 cores.

Design notes:
  - trained on normal rows only (y <= TAIL_THR), like every other mu family;
  - categoricals become embeddings sized by cardinality, with an explicit index 0
    for unseen levels, so a level absent from training is handled at predict time;
  - numerics are quantile-transformed to a bounded range, because the gap columns
    have heavy tails that wreck a plain standardisation, and missing values get a
    zero plus an indicator column;
  - the target is scaled to unit variance so the loss is comparable across runs.
"""
import numpy as np
import pandas as pd

from features import CAT_COLS
from model import TAIL_THR

EMBED_CAP = 32
BATCH = 4096
EPOCHS = 12  # 12 and 20 are equivalent in the blend; 12 is 40% faster
LR = 2e-3
HIDDEN = (256, 128, 64)
CLIP_Q = (0.001, 0.999)


def _embed_dim(n_levels):
    return int(min(EMBED_CAP, round(1.6 * n_levels ** 0.56) + 1))


class _Prep:
    """Fitted on training rows, applied unchanged to anything else."""

    def __init__(self, X):
        self.cats = [c for c in CAT_COLS if c in X.columns]
        self.nums = [c for c in X.columns if c not in self.cats]
        self.levels = {}
        for c in self.cats:
            vals = pd.Index(X[c].astype(str).unique())
            # index 0 is reserved for unseen levels
            self.levels[c] = {v: i + 1 for i, v in enumerate(vals)}
        q = X[self.nums].astype("float32")
        self.lo = q.quantile(CLIP_Q[0])
        self.hi = q.quantile(CLIP_Q[1])
        span = (self.hi - self.lo).replace(0, 1.0)
        self.span = span
        self.med = q.median()

    def codes(self, X):
        out = np.zeros((len(X), len(self.cats)), dtype=np.int64)
        for j, c in enumerate(self.cats):
            m = self.levels[c]
            out[:, j] = X[c].astype(str).map(m).fillna(0).to_numpy(dtype=np.int64)
        return out

    def numeric(self, X):
        q = X[self.nums].astype("float32")
        miss = q.isna().to_numpy(dtype=np.float32)
        q = q.clip(lower=self.lo, upper=self.hi, axis=1)
        z = ((q - self.med) / self.span).fillna(0.0).to_numpy(dtype=np.float32)
        return np.hstack([z, miss])


class _NeuralMu:
    def __init__(self, model, prep, y_mean, y_std, device):
        self.model, self.prep = model, prep
        self.y_mean, self.y_std, self.device = y_mean, y_std, device

    def predict(self, X):
        import torch
        self.model.eval()
        cc = self.prep.codes(X)
        nn_ = self.prep.numeric(X)
        out = np.empty(len(X), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(X), 16384):
                c = torch.from_numpy(cc[i:i + 16384]).to(self.device)
                n = torch.from_numpy(nn_[i:i + 16384]).to(self.device)
                out[i:i + 16384] = self.model(c, n).squeeze(-1).cpu().numpy()
        return np.clip(out * self.y_std + self.y_mean, 0, None)


def _build(prep, n_numeric, device):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([
                nn.Embedding(len(prep.levels[c]) + 1, _embed_dim(len(prep.levels[c])))
                for c in prep.cats])
            width = sum(e.embedding_dim for e in self.embs) + n_numeric
            layers, prev = [], width
            for h in HIDDEN:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(0.1)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, codes, nums):
            parts = [e(codes[:, j]) for j, e in enumerate(self.embs)]
            parts.append(nums)
            return self.mlp(torch.cat(parts, dim=1))

    return Net().to(device)


def train_mu_neural(X, y, epochs=EPOCHS, seed=0, verbose=True):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))

    normal = y <= TAIL_THR
    Xn = X.loc[normal]
    yn = y[normal].astype(np.float32)
    prep = _Prep(Xn)
    codes = prep.codes(Xn)
    nums = prep.numeric(Xn)
    y_mean, y_std = float(yn.mean()), float(yn.std() or 1.0)
    target = ((yn - y_mean) / y_std).astype(np.float32)

    model = _build(prep, nums.shape[1], device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n = len(target)
    steps = int(np.ceil(n / BATCH)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    lossf = nn.MSELoss()
    ct = torch.from_numpy(codes)
    nt = torch.from_numpy(nums)
    tt = torch.from_numpy(target)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            out = model(ct[idx].to(device), nt[idx].to(device)).squeeze(-1)
            loss = lossf(out, tt[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        if verbose:
            print(f"    epoch {ep + 1}/{epochs} train RMSE "
                  f"{np.sqrt(tot / n) * y_std:8.2f}", flush=True)
    return _NeuralMu(model, prep, y_mean, y_std, device)

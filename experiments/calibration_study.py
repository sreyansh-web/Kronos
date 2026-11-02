"""
Calibration study for Kronos: does P(close_up) from the model beat trivial baselines?

No trading, no Polymarket. For each evaluation point t in a historical bar series:
  - condition Kronos on the last L bars (with time stamps),
  - roll out N Monte Carlo paths over H bars,
  - P_model(up) = fraction of paths whose terminal close > last observed close,
  - realized outcome = 1 if the actual close at t+H-1 > close at t-1.

Scored with Brier, log loss, and a reliability table, against:
  - base rate (always predict the training-set up frequency),
  - persistence (sign of the last bar-to-bar return),
  - coin flip (p = 0.5).

Usage:
  python calibration_study.py --csv btc_bars.csv --lookback 128 --horizon 12 --paths 16 --stride 5
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append("..")
from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps

FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def load_bars(path):
    df = pd.read_csv(path)
    if "timestamps" not in df.columns:
        raise ValueError("CSV needs a 'timestamps' column; time stamps condition the model.")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df.reset_index(drop=True)


@torch.no_grad()
def mc_terminal_closes(tokenizer, model, ctx_df, x_ts, y_ts, paths, T, top_p, device):
    """Per-path terminal close. Unlike auto_regressive_inference, nothing is averaged across paths."""
    x = ctx_df[FEATURES].values.astype(np.float32)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    xn = np.clip((x - mu) / (sd + 1e-5), -5, 5)
    x_stamp = calc_time_stamps(x_ts).values.astype(np.float32)
    y_stamp = calc_time_stamps(y_ts).values.astype(np.float32)
    H = len(y_ts)

    x_t = torch.from_numpy(xn).unsqueeze(0).repeat(paths, 1, 1).to(device)
    stamp = torch.from_numpy(np.concatenate([x_stamp, y_stamp])).unsqueeze(0).repeat(paths, 1, 1).to(device)
    L = xn.shape[0]

    pre0, post0 = tokenizer.encode(x_t, half=True)
    pre, post = pre0.clone(), post0.clone()

    for i in range(H):
        cur = L + i
        s1_logits, ctx = model.decode_s1(pre[:, :cur], post[:, :cur], stamp[:, :cur])
        s1 = sample_top_p(s1_logits[:, -1, :], T, top_p)
        s2_logits = model.decode_s2(ctx, s1)
        s2 = sample_top_p(s2_logits[:, -1, :], T, top_p)
        pre = torch.cat([pre, s1.unsqueeze(1)], dim=1)
        post = torch.cat([post, s2.unsqueeze(1)], dim=1)

    z = tokenizer.decode([pre, post], half=True).cpu().numpy()  # (paths, L+H, 6)
    closes = z[:, -1, 3] * (sd[3] + 1e-5) + mu[3]
    return closes


def sample_top_p(logits, T, top_p):
    probs = torch.softmax(logits / T, dim=-1)
    sorted_p, idx = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sorted_p, dim=-1)
    sorted_p[(cum - sorted_p) > top_p] = 0.0
    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_p, 1)
    return idx.gather(-1, choice).squeeze(-1)


def scores(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    brier = np.mean((p - y) ** 2)
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return brier, logloss


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append((f"{lo:.1f}-{hi:.1f}", int(m.sum()), p[m].mean(), y[m].mean()))
    return pd.DataFrame(rows, columns=["bin", "n", "mean_pred", "freq_up"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="NeoQuasar/Kronos-base")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--lookback", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--T", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="calibration_results.csv")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    model = Kronos.from_pretrained(args.model).to(device).eval()

    df = load_bars(args.csv)
    close = df["close"].values
    L, H = args.lookback, args.horizon

    rows = []
    for t in range(L, len(df) - H, args.stride):
        ctx = df.iloc[t - L:t]
        fut_ts = df["timestamps"].iloc[t:t + H]
        closes = mc_terminal_closes(tokenizer, model, ctx, ctx["timestamps"], fut_ts,
                                    args.paths, args.T, args.top_p, device)
        p_model = float(np.mean(closes > close[t - 1]))
        y = float(close[t + H - 1] > close[t - 1])
        persist = float(close[t - 1] > close[t - 2])
        rows.append((df["timestamps"].iloc[t - 1], p_model, persist, y))

    res = pd.DataFrame(rows, columns=["asof", "p_model", "p_persist", "y"])
    res.to_csv(args.out, index=False)

    y = res["y"].values
    base = y.mean()
    print(f"Evaluated {len(res)} points | base rate up = {base:.3f}")
    for name, p in [("model", res["p_model"].values),
                    ("persistence", res["p_persist"].values),
                    ("base_rate", np.full_like(y, base)),
                    ("coin_flip", np.full_like(y, 0.5))]:
        b, ll = scores(p, y)
        print(f"{name:<12} Brier={b:.4f}  LogLoss={ll:.4f}")

    print("\nReliability (model):")
    print(reliability(res["p_model"].values, y).to_string(index=False))
    print(f"\nPer-point results saved to {args.out}")


if __name__ == "__main__":
    main()

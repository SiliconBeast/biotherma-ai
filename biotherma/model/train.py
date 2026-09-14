"""
Train the BioTherma-AI GRU forecaster and export NumPy weights.

Run this on a laptop, not the board. It needs torch; the board does not.

    python train.py --csv baseline.csv --out weights.npz

baseline.csv: ts,air_c,rh,probe_c   -- captured from a stable digester run.
Capture at least an hour. Two is better. Only NORMAL operation goes in here;
the point is that the model has never seen a fault, so faults produce large
residuals. Do not include your hair-dryer demo in the training set.

Export a CSV from the board's SQLite with:
    sqlite3 ~/biotherma.db -header -csv \
      "SELECT ts,air_c,rh,probe_c FROM samples ORDER BY ts" > baseline.csv
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

WINDOW = 24
ALL_FEATURES = ["air_c", "rh", "probe_c"]


class Forecaster(nn.Module):
    def __init__(self, n_feat, hidden=16):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_feat)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def windows(x, w):
    n = x.shape[0] - w
    if n <= 0:
        raise SystemExit(f"need more than {w} samples, got {x.shape[0]}")
    xs = np.stack([x[i:i + w] for i in range(n)])
    ys = x[w:w + n]
    return xs, ys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="weights.npz")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--features", default=",".join(ALL_FEATURES),
                   help="comma-separated subset of air_c,rh,probe_c. Must match "
                        "BIOTHERMA_FEATURES on the board. Use probe_c alone if "
                        "you have no humidity sensor.")
    args = p.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    bad = [f for f in features if f not in ALL_FEATURES]
    if bad:
        raise SystemExit(f"unknown features: {bad}")
    print(f"training on {features}")

    raw = np.genfromtxt(args.csv, delimiter=",", names=True)
    data = np.column_stack([raw[f] for f in features]).astype(np.float64)
    data = data[~np.isnan(data).any(axis=1)]
    print(f"loaded {data.shape[0]} clean samples")

    mu = data.mean(axis=0)
    sigma = data.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    norm = (data - mu) / sigma

    # Chronological split -- shuffling a time series leaks the future.
    cut = int(len(norm) * (1 - args.val_frac))
    xs_tr, ys_tr = windows(norm[:cut], WINDOW)
    xs_va, ys_va = windows(norm[cut:], WINDOW)

    xt = torch.tensor(xs_tr, dtype=torch.float32)
    yt = torch.tensor(ys_tr, dtype=torch.float32)
    xv = torch.tensor(xs_va, dtype=torch.float32)
    yv = torch.tensor(ys_va, dtype=torch.float32)

    model = Forecaster(len(features), args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossfn = nn.MSELoss()

    best, best_state = float("inf"), None
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = lossfn(model(xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = lossfn(model(xv), yv).item()
        if vloss < best:
            best, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 25 == 0:
            print(f"ep {ep:4d}  train {loss.item():.5f}  val {vloss:.5f}")

    model.load_state_dict(best_state)
    print(f"best val loss {best:.5f}")

    # Residual statistics on the validation split, in raw units. These set the
    # z-score scale used on the board -- get them from data the model did not
    # train on, or every threshold will be too tight.
    model.eval()
    with torch.no_grad():
        pv = model(xv).numpy()
    resid = np.abs((pv - ys_va) * sigma)
    resid_mu = resid.mean(axis=0)
    resid_sigma = resid.std(axis=0)
    resid_sigma[resid_sigma < 1e-6] = 1.0
    print("residual mu   ", np.round(resid_mu, 4))
    print("residual sigma", np.round(resid_sigma, 4))

    sd = model.state_dict()
    np.savez(
        args.out,
        W_ih=sd["gru.weight_ih_l0"].numpy().astype(np.float64),
        W_hh=sd["gru.weight_hh_l0"].numpy().astype(np.float64),
        b_ih=sd["gru.bias_ih_l0"].numpy().astype(np.float64),
        b_hh=sd["gru.bias_hh_l0"].numpy().astype(np.float64),
        W_out=sd["head.weight"].numpy().astype(np.float64),
        b_out=sd["head.bias"].numpy().astype(np.float64),
        mu=mu, sigma=sigma,
        resid_mu=resid_mu, resid_sigma=resid_sigma,
        features=np.array(features),
    )
    print(f"wrote {args.out}")

    # Parity check against the NumPy path used on the board.
    from verify import numpy_forward
    ref = model(xv[:1]).detach().numpy()[0] * sigma + mu
    got = numpy_forward(args.out, xs_va[0] * sigma + mu)
    print(f"torch/numpy max abs delta: {np.max(np.abs(ref - got)):.2e}")


if __name__ == "__main__":
    main()

"""Load weights.npz through the same NumPy path the board uses.

Imported by train.py for a parity check, and runnable directly to sanity-check
an exported file:

    python verify.py weights.npz
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "python"))


def _load_forecaster(path):
    # Imported lazily so this file works without the arduino app_utils package
    # installed (i.e. on a laptop).
    import importlib.util
    main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "python", "main.py")
    src = open(main_path).read()
    # Pull only the model section -- main.py imports Bridge at module scope.
    start = src.index("def sigmoid(")
    end = src.index("# ----", start + 10)
    ns = {"np": np}
    exec(compile(src[start:end], "gru", "exec"), ns)
    return ns["GruForecaster"](path)


def numpy_forward(weights_path, window_raw):
    """window_raw: (WINDOW, F) in raw units. Returns next-sample prediction."""
    return _load_forecaster(weights_path).predict(np.asarray(window_raw))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "weights.npz"
    f = _load_forecaster(path)
    print(f"hidden={f.hidden} features={f.n_feat}")
    print(f"mu={np.round(f.mu, 3)}")
    print(f"sigma={np.round(f.sigma, 3)}")
    print(f"resid_mu={np.round(f.resid_mu, 4)}")
    print(f"resid_sigma={np.round(f.resid_sigma, 4)}")
    demo = np.tile(f.mu, (24, 1))
    print(f"prediction at mean input: {np.round(f.predict(demo), 3)}")

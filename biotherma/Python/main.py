"""
BioTherma-AI -- MPU layer (Qualcomm QRB2210 / Debian).

Two things run here.

1. A GRU forecaster over the recent thermal/humidity sequence. It learns the
   normal joint dynamics of the chamber during stable operation and scores the
   one-step forecast residual. A sustained residual means the reactor is no
   longer behaving the way it did during training -- heater fault, lid seal,
   or a stalled culture. This is anomaly detection, not gas sensing.

2. A process model (digester.py) that estimates biogas output, harvest timing,
   carbon-to-nitrogen balance and energy value from the operator's feed log and
   the measured slurry temperature, using published yield coefficients and
   first-order hydrolysis kinetics.

Neither component senses methane. There is no gas sensor in the build; the
yield figures are modelled from feed mass and measured temperature, and the
dashboard says so.

Inference is pure NumPy so nothing has to be compiled on the board.
"""

import json
import os
import sqlite3
import threading
import time
from collections import deque

import numpy as np

from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI

import digester

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "..", "model", "weights.npz")
DB_PATH = os.path.join(os.path.expanduser("~"), "biotherma.db")

WINDOW = 24                 # samples fed to the GRU (24 x 2s = 48s of history)
SENTINEL = -2147483648      # INT32_MIN from the sketch == NaN

ALL_FEATURES = ["air_c", "rh", "probe_c"]
# Default build has no thermistor: a divider needs a physical resistor to
# ground and there is no software substitute. The AM2320's own temperature
# reading becomes the process temperature. Add probe_c back here (and set
# HAS_NTC 1 in the sketch) if you fit the divider later.
DEFAULT_FEATURES = "air_c,rh"
FEATURES = [f for f in
            os.environ.get("BIOTHERMA_FEATURES", DEFAULT_FEATURES).split(",")
            if f.strip() in ALL_FEATURES] or ["air_c", "rh"]

MESO_LOW_C = digester.MESO_LOW_C
MESO_HIGH_C = digester.MESO_HIGH_C

# Residual z-score thresholds. Set these from the residual sigmas train.py
# prints; the defaults are placeholders.
Z_WARN = 2.5
Z_ALERT = 4.0

DEBOUNCE = 5                # consecutive samples before a state change sticks

# Default energy price per kWh-equivalent. Settable from the dashboard.
DEFAULT_PRICE = 0.11
CURRENCY = "CAD"

PROCESS_PERIOD_S = 10.0     # how often the batch model is integrated + pushed

# Status codes sent to the MCU, which owns the colour mapping. The UNO Q's two
# MCU-side onboard RGB LEDs are used: LED3 shows what the reactor is doing,
# LED4 shows whether the readings can be trusted. No external LEDs.
P_IDLE, P_PRODUCING, P_BALANCE, P_HARVEST = 0, 1, 2, 3
H_IDLE, H_OK, H_WATCH, H_STALL, H_BAND = 0, 1, 2, 3, 4

PROCESS_CODE = {
    "PRODUCING": P_PRODUCING,
    "BALANCE":   P_BALANCE,
    "HARVEST":   P_HARVEST,
    "WATCH":     P_PRODUCING,   # drifting, but the batch is still running
    "STALL":     P_PRODUCING,
    "BAND":      P_PRODUCING,
    "EMPTY":     P_IDLE,
    "INIT":      P_IDLE,
}

HEALTH_CODE = {
    "PRODUCING": H_OK,
    "BALANCE":   H_OK,
    "HARVEST":   H_OK,
    "WATCH":     H_WATCH,
    "STALL":     H_STALL,
    "BAND":      H_BAND,
    "EMPTY":     H_IDLE,
    "INIT":      H_IDLE,
}


# --------------------------------------------------------------------------
# NumPy GRU
# --------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class GruForecaster:
    """Single-layer GRU + linear head, PyTorch weight layout.

    Expects an npz containing:
      W_ih (3H, F), W_hh (3H, H), b_ih (3H,), b_hh (3H,)
      W_out (F, H), b_out (F,)
      mu (F,), sigma (F,)                feature normalisation from training
      resid_mu (F,), resid_sigma (F,)    residual stats from training
    Gate order is r, z, n -- same as torch.nn.GRU.
    """

    def __init__(self, path, z_alert=4.0):
        self.z_alert = z_alert
        d = np.load(path)
        self.W_ih, self.W_hh = d["W_ih"], d["W_hh"]
        self.b_ih, self.b_hh = d["b_ih"], d["b_hh"]
        self.W_out, self.b_out = d["W_out"], d["b_out"]
        self.mu, self.sigma = d["mu"], d["sigma"]
        self.resid_mu, self.resid_sigma = d["resid_mu"], d["resid_sigma"]
        self.hidden = self.W_hh.shape[1]
        self.n_feat = self.mu.shape[0]

    def _cell(self, x, h):
        gi = self.W_ih @ x + self.b_ih
        gh = self.W_hh @ h + self.b_hh
        H = self.hidden
        r = sigmoid(gi[:H] + gh[:H])
        z = sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        return (1.0 - z) * n + z * h

    def predict(self, seq):
        """seq: (WINDOW, F) raw units. Returns predicted next sample, raw units."""
        x = (np.asarray(seq, dtype=np.float64) - self.mu) / self.sigma
        h = np.zeros(self.hidden)
        for t in range(x.shape[0]):
            h = self._cell(x[t], h)
        y = self.W_out @ h + self.b_out
        return y * self.sigma + self.mu

    def score(self, predicted, actual):
        """Per-feature z-scores of the residual, plus a scalar index 0-100."""
        resid = np.abs(np.asarray(actual) - np.asarray(predicted))
        z = (resid - self.resid_mu) / np.maximum(self.resid_sigma, 1e-6)
        peak = float(np.max(z))
        index = float(np.clip(peak / self.z_alert, 0.0, 1.0) * 100.0)
        return z, peak, index


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

class Store:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS samples (
                ts REAL PRIMARY KEY, air_c REAL, rh REAL, probe_c REAL,
                pred REAL, z_peak REAL, idx REAL, state TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_ts ON samples(ts);

            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started REAL, harvested REAL,
                conversion REAL DEFAULT 0, last_update REAL,
                peak_rate REAL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS feeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER, ts REAL, kind TEXT, kg REAL
            );
            CREATE TABLE IF NOT EXISTS settings (k TEXT PRIMARY KEY, v TEXT);
        """)
        self.conn.commit()

    # -- samples ---------------------------------------------------------
    def insert_sample(self, row):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?,?,?,?)", row)
            self.conn.commit()

    def recent(self, limit=720):
        with self.lock:
            cur = self.conn.execute(
                "SELECT ts, air_c, rh, probe_c, idx, state FROM samples "
                "ORDER BY ts DESC LIMIT ?", (limit,))
            return list(reversed(cur.fetchall()))

    # -- settings --------------------------------------------------------
    def get_setting(self, key, default=None):
        with self.lock:
            cur = self.conn.execute("SELECT v FROM settings WHERE k=?", (key,))
            row = cur.fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO settings VALUES (?,?)",
                              (key, json.dumps(value)))
            self.conn.commit()

    # -- batches ---------------------------------------------------------
    def open_batch(self):
        with self.lock:
            cur = self.conn.execute(
                "SELECT id, started, conversion, last_update, peak_rate FROM batches "
                "WHERE harvested IS NULL ORDER BY id DESC LIMIT 1")
            return cur.fetchone()

    def create_batch(self, started):
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO batches (started, last_update) VALUES (?,?)",
                (started, started))
            self.conn.commit()
            return cur.lastrowid

    def close_batch(self, batch_id, ts):
        with self.lock:
            self.conn.execute("UPDATE batches SET harvested=? WHERE id=?", (ts, batch_id))
            self.conn.commit()

    def save_batch_state(self, batch_id, conversion, last_update, peak_rate):
        with self.lock:
            self.conn.execute(
                "UPDATE batches SET conversion=?, last_update=?, peak_rate=? WHERE id=?",
                (conversion, last_update, peak_rate, batch_id))
            self.conn.commit()

    def add_feed(self, batch_id, ts, kind, kg):
        with self.lock:
            self.conn.execute(
                "INSERT INTO feeds (batch_id, ts, kind, kg) VALUES (?,?,?,?)",
                (batch_id, ts, kind, kg))
            self.conn.commit()

    def feeds_for(self, batch_id):
        with self.lock:
            cur = self.conn.execute(
                "SELECT ts, kind, kg FROM feeds WHERE batch_id=? ORDER BY ts", (batch_id,))
            return cur.fetchall()


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

class BioTherma:
    def __init__(self):
        self.store = Store(DB_PATH)
        self.window = deque(maxlen=WINDOW)
        self.model = None
        self.state = "INIT"
        self.breach = 0
        self.clear = 0
        self.last = {}
        self.mcu_errors = 0
        self.array_on = True
        self.price = self.store.get_setting("price", DEFAULT_PRICE)
        self.last_process = 0.0
        self.lock = threading.RLock()

        if os.path.exists(WEIGHTS_PATH):
            try:
                self.model = GruForecaster(WEIGHTS_PATH, z_alert=Z_ALERT)
                print(f"[biotherma] model loaded, hidden={self.model.hidden}")
                if self.model.n_feat != len(FEATURES):
                    print(f"[biotherma] WEIGHTS MISMATCH: weights.npz has "
                          f"{self.model.n_feat} features, BIOTHERMA_FEATURES has "
                          f"{len(FEATURES)} ({','.join(FEATURES)}). Retrain or fix "
                          f"the env var -- inference disabled.")
                    self.model = None
            except Exception as e:
                print(f"[biotherma] model load failed: {e}")
        else:
            print("[biotherma] no weights.npz -- running in baseline-capture mode")

        self.batch_id, self.batch = self._restore_batch()

    # -- batch lifecycle -------------------------------------------------
    def _restore_batch(self):
        row = self.store.open_batch()
        if row is None:
            started = time.time()
            bid = self.store.create_batch(started)
            print(f"[biotherma] started batch {bid}")
            return bid, digester.Batch(started=started)

        bid, started, conversion, last_update, peak_rate = row
        b = digester.Batch(started=started)
        for ts, kind, kg in self.store.feeds_for(bid):
            try:
                b.add_feed(kind, kg, ts=ts)
            except ValueError as e:
                print(f"[biotherma] skipping bad feed row: {e}")
        # Restore integration state after the feeds, since add_feed rescales it.
        b.conversion = conversion or 0.0
        b.last_update = last_update or started
        b.peak_rate_m3_day = peak_rate or 0.0
        print(f"[biotherma] resumed batch {bid}: {b.total_kg():.1f} kg, "
              f"{b.conversion * 100:.0f}% converted")
        return bid, b

    def harvest(self):
        with self.lock:
            now = time.time()
            self.store.save_batch_state(self.batch_id, self.batch.conversion,
                                        self.batch.last_update,
                                        self.batch.peak_rate_m3_day)
            self.store.close_batch(self.batch_id, now)
            self.batch_id = self.store.create_batch(now)
            self.batch = digester.Batch(started=now)
            print(f"[biotherma] harvested; new batch {self.batch_id}")

    def add_feed(self, kind, kg):
        with self.lock:
            feed = self.batch.add_feed(kind, kg)
            self.store.add_feed(self.batch_id, feed.ts, feed.kind, feed.kg)
            return feed

    # -- LED policy ------------------------------------------------------
    def leds_for(self, state):
        """(process code, health code) for the onboard RGB pair."""
        return (PROCESS_CODE.get(state, P_IDLE), HEALTH_CODE.get(state, H_IDLE))

    def push_leds(self):
        process, health = self.leds_for(self.state)
        try:
            Bridge.call("set_status", int(process), int(health),
                        1 if self.array_on else 0)
        except Exception as e:
            print(f"[biotherma] bridge set_status failed: {e}")

    # -- classification --------------------------------------------------
    def classify(self, temp_c, z_peak):
        # Hard faults first. A thermal excursion outranks anything the model says.
        if temp_c is not None and not (MESO_LOW_C <= temp_c <= MESO_HIGH_C):
            return "BAND"

        if z_peak is not None:
            if z_peak >= Z_ALERT:
                return "STALL"
            if z_peak >= Z_WARN:
                return "WATCH"
        elif self.model is not None:
            return "INIT"          # window still filling

        # Nominal. Now report on the process rather than the sensors.
        if not self.batch.feeds:
            return "EMPTY"
        if self.batch.ready_to_harvest():
            return "HARVEST"
        if self.batch.cn_advice()["status"] in ("low", "high"):
            return "BALANCE"
        return "PRODUCING"

    def transition(self, candidate):
        """Debounce escalation and recovery. BAND is immediate -- a temperature
        excursion is not something to sit on for ten seconds."""
        if candidate == self.state:
            self.breach = self.clear = 0
            return False

        if candidate == "BAND" or self.state == "INIT":
            self.state = candidate
            self.breach = self.clear = 0
            return True

        rank = {"EMPTY": 0, "PRODUCING": 0, "BALANCE": 1, "HARVEST": 1,
                "WATCH": 2, "STALL": 3, "BAND": 4}
        if rank.get(candidate, 0) > rank.get(self.state, 0):
            self.breach += 1
            self.clear = 0
            if self.breach >= DEBOUNCE:
                self.state = candidate
                self.breach = 0
                return True
        else:
            self.clear += 1
            self.breach = 0
            if self.clear >= DEBOUNCE:
                self.state = candidate
                self.clear = 0
                return True
        return False

    # -- telemetry ingress ----------------------------------------------
    def on_telemetry(self, air_x100, rh_x100, probe_x100, err_count):
        def dec(v):
            return None if v == SENTINEL else v / 100.0

        air, rh, probe = dec(air_x100), dec(rh_x100), dec(probe_x100)
        self.mcu_errors = int(err_count)
        ts = time.time()
        channels = {"air_c": air, "rh": rh, "probe_c": probe}

        # Process temperature: the thermistor if one is fitted, otherwise the
        # AM2320. Everything downstream -- band check, kinetics, harvest timing
        # -- keys off this one number.
        temp = probe if probe is not None else air

        missing = [f for f in FEATURES if channels[f] is None]
        if temp is None and "air_c" not in missing:
            missing.append("air_c")
        if missing:
            self.last = {"ts": ts, "air_c": air, "rh": rh, "probe_c": probe,
                         "state": self.state, "index": 0.0,
                         "mcu_errors": self.mcu_errors,
                         "fault": "sensor", "missing": missing}
            ui.emit("sample", self.last)
            return

        features = [channels[f] for f in FEATURES]
        pred = z_peak = index = None
        if self.model is not None and len(self.window) == WINDOW:
            predicted = self.model.predict(np.array(self.window))
            _, z_peak, index = self.model.score(predicted, features)
            pred = float(predicted[0])
        self.window.append(features)

        # Integrate the batch against the measured temperature.
        with self.lock:
            self.batch.update(temp, now=ts)

        candidate = self.classify(temp, z_peak)
        if self.transition(candidate):
            self.push_leds()
            z_txt = f"{z_peak:.2f}" if z_peak is not None else "n/a"
            print(f"[biotherma] state -> {self.state} (z={z_txt})")

        self.store.insert_sample((ts, air, rh, probe, pred,
                                  z_peak or 0.0, index or 0.0, self.state))

        self.last = {
            "ts": ts,
            "air_c": round(air, 2) if air is not None else None,
            "rh": round(rh, 2) if rh is not None else None,
            "probe_c": round(probe, 2) if probe is not None else None,
            "temp_c": round(temp, 2),
            "temp_source": "probe" if probe is not None else "air",
            "z_peak": round(z_peak, 3) if z_peak is not None else None,
            "index": round(index, 1) if index is not None else 0.0,
            "state": self.state,
            "mcu_errors": self.mcu_errors,
            "warming_up": self.model is not None and len(self.window) < WINDOW,
            "modelled": self.model is not None,
        }
        ui.emit("sample", self.last)

        if ts - self.last_process >= PROCESS_PERIOD_S:
            self.last_process = ts
            self.push_process(temp)

    def push_process(self, probe_c):
        with self.lock:
            snap = self.batch.snapshot(probe_c, self.price, CURRENCY)
            self.store.save_batch_state(self.batch_id, self.batch.conversion,
                                        self.batch.last_update,
                                        self.batch.peak_rate_m3_day)
        snap["batch_id"] = self.batch_id
        snap["price"] = self.price
        ui.emit("process", snap)
        return snap


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

app = BioTherma()
ui = WebUI()


def _probe():
    """Current process temperature, whichever sensor is providing it."""
    return app.last.get("temp_c")


@ui.on("history")
def on_history(_=None):
    rows = app.store.recent(720)
    ui.emit("history", [
        {"ts": r[0], "air_c": r[1], "rh": r[2], "probe_c": r[3],
         "index": r[4], "state": r[5]} for r in rows
    ])
    ui.emit("feedstocks", [
        {"kind": k, "label": v["label"], "cn": v["cn"],
         "yield_m3_per_kg": v["biogas_m3_per_kg"]}
        for k, v in sorted(digester.FEEDSTOCKS.items(), key=lambda kv: kv[1]["label"])
    ])
    app.push_process(_probe())


@ui.on("add_feed")
def on_add_feed(payload):
    try:
        feed = app.add_feed(payload["kind"], float(payload["kg"]))
    except (KeyError, ValueError, TypeError) as e:
        ui.emit("error", {"where": "add_feed", "message": str(e)})
        return
    print(f"[biotherma] fed {feed.kg} kg {feed.kind}")
    snap = app.push_process(_probe())
    # Re-evaluate immediately so the array reflects the new blend.
    if app.transition(app.classify(_probe(), app.last.get("z_peak"))):
        app.push_leds()
    ui.emit("fed", {"feed": feed.as_dict(), "cn": snap["cn"]})


@ui.on("harvest")
def on_harvest(_=None):
    app.harvest()
    app.state = "INIT"
    app.push_leds()
    app.push_process(_probe())


@ui.on("set_price")
def on_set_price(payload):
    try:
        price = float(payload["price"])
    except (KeyError, ValueError, TypeError):
        return
    if price < 0:
        return
    app.price = price
    app.store.set_setting("price", price)
    app.push_process(_probe())


@ui.on("set_array")
def on_set_array(payload):
    app.array_on = bool(payload.get("on", True))
    app.push_leds()
    ui.emit("array", {"on": app.array_on})


@ui.on("lamp_test")
def on_lamp_test(_=None):
    """The sketch owns the sequence -- it knows the channel pins."""
    try:
        Bridge.call("lamp_test")
    except Exception as e:
        print(f"[biotherma] bridge lamp_test failed: {e}")


Bridge.provide("telemetry", app.on_telemetry)


def heartbeat():
    """If the MCU stops publishing, go dark rather than showing a stale green.
    A monitoring device that lies when it dies is worse than one that is
    obviously off."""
    while True:
        time.sleep(15)
        last_ts = app.last.get("ts", 0)
        if last_ts and time.time() - last_ts > 15 and app.state != "INIT":
            app.state = "INIT"
            app.push_leds()
            ui.emit("sample", {"state": "INIT", "fault": "mcu_silent"})


threading.Thread(target=heartbeat, daemon=True).start()

if __name__ == "__main__":
    app.push_leds()
    App.run()

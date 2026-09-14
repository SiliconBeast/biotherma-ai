"""
Bench simulator -- drives the BioTherma pipeline with no hardware attached.

This exists so UI, model and state-machine work can continue while parts are in
transit. Output from this script is SIMULATED. Anything you publish that was
produced here must say so; do not pass a sim run off as a captured measurement.

    python sim.py                      # normal chamber, prints state changes
    python sim.py --fault heater --at 300
    python sim.py --fault lid --at 300
    python sim.py --fault stall --at 300
    python sim.py --hours 2 --csv baseline_sim.csv   # write a training CSV
    python sim.py --realtime           # 2s cadence, for driving the dashboard

Physics, such as it is: first-order thermal lag of the slurry toward a heater
setpoint, ambient coupling through the wall with a diurnal swing, headspace air
sitting between slurry and ambient, and headspace RH pinned near saturation
with small excursions when the lid opens. It is a plausible-looking chamber,
not a validated model of one.
"""

import argparse
import csv
import math
import sys
import time

import numpy as np

DT = 2.0  # seconds per sample, matches SAMPLE_PERIOD_MS in the sketch


class Chamber:
    def __init__(self, seed=0, setpoint=34.0, ambient=19.0):
        self.rng = np.random.default_rng(seed)
        self.setpoint = setpoint
        self.ambient_base = ambient
        self.slurry = setpoint - 0.4
        self.air = setpoint - 2.0
        self.rh = 97.5
        self.t = 0.0

        self.heater_ok = True
        self.lid_open = False
        self.stalled = False

        self.tau_slurry = 900.0    # s, slurry thermal time constant
        self.tau_air = 120.0
        self.k_wall = 0.06         # ambient coupling
        self.bio_watts = 0.010     # degC/s of biological self-heating, small

    def ambient(self):
        # Slow diurnal swing plus drift.
        return self.ambient_base + 3.2 * math.sin(2 * math.pi * self.t / 86400.0)

    def step(self):
        self.t += DT
        amb = self.ambient()

        drive = 0.0
        if self.heater_ok and self.slurry < self.setpoint:
            drive = (self.setpoint - self.slurry) / self.tau_slurry * 40.0

        bio = 0.0 if self.stalled else self.bio_watts
        loss = self.k_wall * (self.slurry - amb) / self.tau_slurry * 60.0
        if self.lid_open:
            loss *= 4.0

        self.slurry += (drive + bio - loss) * DT
        self.slurry += self.rng.normal(0, 0.004)

        target_air = 0.6 * self.slurry + 0.4 * (amb if self.lid_open else self.slurry - 1.5)
        self.air += (target_air - self.air) * (DT / self.tau_air) * 10.0
        self.air += self.rng.normal(0, 0.02)

        # Headspace stays near saturation; opening the lid vents it briefly.
        rh_target = 88.0 if self.lid_open else 97.8
        self.rh += (rh_target - self.rh) * 0.05
        self.rh = float(np.clip(self.rh + self.rng.normal(0, 0.12), 0, 100))

        return self.air, self.rh, self.slurry


FAULTS = {
    "heater": "heater contactor fails, slurry drifts toward ambient",
    "lid":    "lid seal opens, headspace vents and heat loss quadruples",
    "stall":  "culture stops producing, biological self-heating goes to zero",
    "none":   "no fault",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=0.5)
    p.add_argument("--fault", choices=sorted(FAULTS), default="none")
    p.add_argument("--at", type=int, default=600, help="sample index the fault starts")
    p.add_argument("--csv", help="write samples here instead of feeding the pipeline")
    p.add_argument("--realtime", action="store_true", help="sleep DT between samples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n = int(args.hours * 3600 / DT)
    ch = Chamber(seed=args.seed)

    print(f"[sim] SIMULATED DATA -- {n} samples, {args.hours}h, "
          f"fault={args.fault} ({FAULTS[args.fault]})", file=sys.stderr)

    sink = None
    if args.csv:
        fh = open(args.csv, "w", newline="")
        sink = csv.writer(fh)
        sink.writerow(["ts", "air_c", "rh", "probe_c"])
        if args.fault != "none":
            print("[sim] warning: writing a faulted run to CSV. Train only on "
                  "clean runs or the model learns the fault as normal.",
                  file=sys.stderr)
    else:
        app = load_pipeline()

    t0 = time.time()
    for i in range(n):
        if i == args.at and args.fault != "none":
            if args.fault == "heater":
                ch.heater_ok = False
            elif args.fault == "lid":
                ch.lid_open = True
            elif args.fault == "stall":
                ch.stalled = True
            print(f"[sim] t={i * DT:.0f}s  injecting {args.fault}", file=sys.stderr)

        air, rh, probe = ch.step()

        if sink:
            sink.writerow([f"{t0 + i * DT:.3f}", f"{air:.3f}", f"{rh:.3f}", f"{probe:.3f}"])
        else:
            app.on_telemetry(int(round(air * 100)), int(round(rh * 100)),
                             int(round(probe * 100)), 0)
        if args.realtime:
            time.sleep(DT)

    if sink:
        fh.close()
        print(f"[sim] wrote {args.csv}", file=sys.stderr)
    else:
        print(f"[sim] final state: {app.state}  probe={probe:.2f}C  rh={rh:.1f}%",
              file=sys.stderr)


def load_pipeline():
    """Import main.py with the App Lab runtime stubbed out."""
    import importlib.util
    import os
    import types

    for name in ("arduino", "arduino.app_utils", "arduino.app_bricks",
                 "arduino.app_bricks.web_ui"):
        sys.modules.setdefault(name, types.ModuleType(name))

    au = sys.modules["arduino.app_utils"]
    au.App = type("App", (), {"run": staticmethod(lambda: None)})
    au.Bridge = type("Bridge", (), {
        "call": staticmethod(lambda *a, **k: None),
        "provide": staticmethod(lambda *a, **k: None),
    })

    class WebUI:
        def send_message(self, *a, **k):
            pass

        def on_message(self, _name, _handler):
            pass

    sys.modules["arduino.app_bricks.web_ui"].WebUI = WebUI

    here = os.path.dirname(os.path.abspath(__file__))
    pydir = os.path.join(here, "..", "python")
    if pydir not in sys.path:
        sys.path.insert(0, pydir)
    spec = importlib.util.spec_from_file_location(
        "biotherma_main", os.path.join(here, "..", "python", "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app


if __name__ == "__main__":
    main()

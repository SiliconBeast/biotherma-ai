"""
BioTherma-AI -- digester process model.

Estimates biogas output, harvest timing, carbon-to-nitrogen balance and energy
value for a small batch anaerobic digester.

This is a physics-and-literature model, not a gas measurement. It takes two
real inputs -- what the operator fed the reactor, and the slurry temperature
measured by the thermistor -- and applies published yield coefficients and
first-order hydrolysis kinetics to them. Every constant below is sourced and
adjustable. Nothing here claims to sense methane.

Kinetics
--------
Batch biogas release follows first-order hydrolysis:

    B(t) = B0 * (1 - exp(-k*t))

with k temperature-corrected by the modified Arrhenius form used throughout
the wastewater literature:

    k(T) = k35 * theta^(T - 35)

theta = 1.08 for mesophilic digestion. Below MESO_LOW the culture slows
sharply; above MESO_HIGH mesophiles are inhibited and k is penalised rather
than extrapolated upward.
"""

import math
import time

# --------------------------------------------------------------------------
# Feedstock coefficients
#
# biogas_m3_per_kg is per kilogram of WET mass as fed, already accounting for
# typical total-solids and volatile-solids fractions. Ranges in the literature
# are wide; these are mid-range values for domestic-scale digestion.
#
# Sources: Angelidaki & Sanders (2004); Zhang et al. (2007), food waste BMP;
# Cornell Waste Management Institute C:N tables; IPCC 2006 Vol.5 Ch.4.
# --------------------------------------------------------------------------

FEEDSTOCKS = {
    "food_scraps":   {"biogas_m3_per_kg": 0.105, "ch4_frac": 0.62, "cn": 17,
                      "label": "Mixed food scraps"},
    "fruit_veg":     {"biogas_m3_per_kg": 0.075, "ch4_frac": 0.58, "cn": 33,
                      "label": "Fruit and vegetable trim"},
    "cooked_grains": {"biogas_m3_per_kg": 0.140, "ch4_frac": 0.63, "cn": 22,
                      "label": "Cooked rice, pasta, bread"},
    "dairy":         {"biogas_m3_per_kg": 0.180, "ch4_frac": 0.68, "cn": 14,
                      "label": "Dairy and fats"},
    "coffee_grounds": {"biogas_m3_per_kg": 0.090, "ch4_frac": 0.60, "cn": 20,
                       "label": "Coffee grounds"},
    "grass":         {"biogas_m3_per_kg": 0.085, "ch4_frac": 0.55, "cn": 19,
                      "label": "Grass clippings"},
    "leaves":        {"biogas_m3_per_kg": 0.045, "ch4_frac": 0.52, "cn": 60,
                      "label": "Dry leaves"},
    "cardboard":     {"biogas_m3_per_kg": 0.060, "ch4_frac": 0.55, "cn": 350,
                      "label": "Shredded cardboard or paper"},
    "manure":        {"biogas_m3_per_kg": 0.040, "ch4_frac": 0.60, "cn": 20,
                      "label": "Animal manure"},
}

# Nitrogen content is what actually mixes when you blend feedstocks, so C:N
# ratios are combined on a nitrogen-mass basis, not averaged directly.
# Assumed carbon fraction of wet mass, used only to weight the blend.
C_FRAC_WET = 0.12

# --------------------------------------------------------------------------
# Process constants
# --------------------------------------------------------------------------

K35_PER_DAY = 0.18        # first-order hydrolysis rate at 35 C
THETA = 1.08              # modified Arrhenius temperature coefficient
MESO_LOW_C = 30.0
MESO_HIGH_C = 38.0
INHIBIT_ABOVE_C = 40.0    # mesophile die-off begins
FLOOR_C = 15.0            # below this, treat activity as arrested

CH4_MJ_PER_M3 = 35.8      # lower heating value of pure methane
MJ_PER_KWH = 3.6

CN_MIN, CN_MAX = 20.0, 30.0   # healthy operating window
CN_TARGET = 25.0

HARVEST_FRACTION = 0.85   # batch considered spent at 85% of theoretical yield

# Fugitive methane avoided. IPCC GWP100 for fossil methane, AR6.
CH4_KG_PER_M3 = 0.716     # density at STP
GWP100_CH4 = 29.8


class Feed:
    __slots__ = ("ts", "kind", "kg")

    def __init__(self, ts, kind, kg):
        self.ts = ts
        self.kind = kind
        self.kg = float(kg)

    def as_dict(self):
        f = FEEDSTOCKS[self.kind]
        return {"ts": self.ts, "kind": self.kind, "label": f["label"],
                "kg": round(self.kg, 2)}


def rate_constant(temp_c):
    """Temperature-corrected hydrolysis rate, per day. Returns 0 when arrested."""
    if temp_c is None or temp_c < FLOOR_C:
        return 0.0
    if temp_c > INHIBIT_ABOVE_C:
        # Past mesophilic tolerance, activity collapses rather than rising.
        excess = temp_c - INHIBIT_ABOVE_C
        return max(0.0, K35_PER_DAY * (THETA ** (INHIBIT_ABOVE_C - 35.0))
                   * math.exp(-0.35 * excess))
    return K35_PER_DAY * (THETA ** (temp_c - 35.0))


def activity_index(temp_c):
    """0-100, relative to the rate at the middle of the mesophilic window."""
    ref = rate_constant((MESO_LOW_C + MESO_HIGH_C) / 2.0)
    if ref <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * rate_constant(temp_c) / ref))


class Batch:
    """One digester charge. Tracks feed, integrated conversion, and output.

    Conversion is integrated numerically against the measured temperature
    rather than solved in closed form, because k changes with every sample.
    """

    def __init__(self, started=None):
        self.started = started or time.time()
        self.feeds = []
        self.conversion = 0.0       # fraction of theoretical yield released
        self.last_update = self.started
        self.peak_rate_m3_day = 0.0
        self.rate_m3_day = 0.0

    # -- feedstock -------------------------------------------------------
    def add_feed(self, kind, kg, ts=None):
        if kind not in FEEDSTOCKS:
            raise ValueError(f"unknown feedstock {kind!r}")
        if kg <= 0:
            raise ValueError("feed mass must be positive")
        # New material resets conversion proportionally: undigested mass is
        # diluted back into the batch.
        prior = self.theoretical_biogas_m3()
        self.feeds.append(Feed(ts or time.time(), kind, kg))
        total = self.theoretical_biogas_m3()
        if total > 0:
            self.conversion *= (prior / total)
        return self.feeds[-1]

    def total_kg(self):
        return sum(f.kg for f in self.feeds)

    def theoretical_biogas_m3(self):
        return sum(FEEDSTOCKS[f.kind]["biogas_m3_per_kg"] * f.kg for f in self.feeds)

    def ch4_fraction(self):
        """Mass-weighted methane fraction of the blend."""
        num = den = 0.0
        for f in self.feeds:
            s = FEEDSTOCKS[f.kind]
            v = s["biogas_m3_per_kg"] * f.kg
            num += v * s["ch4_frac"]
            den += v
        return (num / den) if den > 0 else 0.0

    def carbon_nitrogen(self):
        """Blended C:N on a nitrogen-mass basis. None if the batch is empty."""
        c_mass = n_mass = 0.0
        for f in self.feeds:
            cn = FEEDSTOCKS[f.kind]["cn"]
            c = f.kg * C_FRAC_WET
            c_mass += c
            n_mass += c / cn
        if n_mass <= 0:
            return None
        return c_mass / n_mass

    # -- integration -----------------------------------------------------
    def update(self, temp_c, now=None):
        now = now or time.time()
        dt_days = max(0.0, (now - self.last_update) / 86400.0)
        self.last_update = now
        if dt_days == 0.0 or not self.feeds:
            return

        k = rate_constant(temp_c)
        remaining = 1.0 - self.conversion
        released = remaining * (1.0 - math.exp(-k * dt_days))
        self.conversion = min(1.0, self.conversion + released)

        theo = self.theoretical_biogas_m3()
        self.rate_m3_day = k * remaining * theo
        self.peak_rate_m3_day = max(self.peak_rate_m3_day, self.rate_m3_day)

    # -- outputs ---------------------------------------------------------
    def biogas_m3(self):
        return self.conversion * self.theoretical_biogas_m3()

    def methane_m3(self):
        return self.biogas_m3() * self.ch4_fraction()

    def energy_kwh(self):
        return self.methane_m3() * CH4_MJ_PER_M3 / MJ_PER_KWH

    def co2e_avoided_kg(self):
        """Fugitive methane that would have vented from landfill, as CO2e.

        Assumes the same mass would otherwise decompose anaerobically in a
        landfill without gas capture. That is the pessimistic-landfill case;
        a captured or composted baseline gives a smaller number.
        """
        return self.methane_m3() * CH4_KG_PER_M3 * GWP100_CH4

    def days_to_harvest(self, temp_c):
        """Days until HARVEST_FRACTION of theoretical yield is released."""
        if not self.feeds or self.conversion >= HARVEST_FRACTION:
            return 0.0
        k = rate_constant(temp_c)
        if k <= 0:
            return float("inf")
        remaining_frac = (1.0 - HARVEST_FRACTION) / (1.0 - self.conversion)
        return -math.log(max(remaining_frac, 1e-9)) / k

    def ready_to_harvest(self):
        return bool(self.feeds) and self.conversion >= HARVEST_FRACTION

    def cn_advice(self):
        """What to add to pull the blend back into the 20-30:1 window."""
        cn = self.carbon_nitrogen()
        if cn is None:
            return {"cn": None, "status": "empty", "message": "Nothing in the reactor yet."}

        if CN_MIN <= cn <= CN_MAX:
            return {"cn": round(cn, 1), "status": "ok",
                    "message": f"Blend is at {cn:.0f}:1, inside the 20–30:1 window."}

        c_mass = self.total_kg() * C_FRAC_WET
        n_mass = c_mass / cn

        if cn < CN_MIN:
            # Nitrogen-rich: needs carbon. Solve for kg of cardboard to reach target.
            src = "cardboard"
            src_cn = FEEDSTOCKS[src]["cn"]
            # (c_mass + x*C_FRAC) / (n_mass + x*C_FRAC/src_cn) = CN_TARGET
            num = CN_TARGET * n_mass - c_mass
            den = C_FRAC_WET * (1.0 - CN_TARGET / src_cn)
            kg = max(0.0, num / den) if den != 0 else 0.0
            return {"cn": round(cn, 1), "status": "low", "add": src, "add_kg": round(kg, 1),
                    "message": f"Blend is nitrogen-heavy at {cn:.0f}:1. Souring risk. "
                               f"Add about {kg:.1f} kg of shredded cardboard."}

        src = "food_scraps"
        src_cn = FEEDSTOCKS[src]["cn"]
        num = c_mass - CN_TARGET * n_mass
        den = C_FRAC_WET * (CN_TARGET / src_cn - 1.0)
        kg = max(0.0, num / den) if den != 0 else 0.0
        return {"cn": round(cn, 1), "status": "high", "add": src, "add_kg": round(kg, 1),
                "message": f"Blend is carbon-heavy at {cn:.0f}:1. Gas production will "
                           f"lag. Add about {kg:.1f} kg of food scraps."}

    def snapshot(self, temp_c, energy_price, currency="CAD"):
        kwh = self.energy_kwh()
        d2h = self.days_to_harvest(temp_c)
        return {
            "feeds": [f.as_dict() for f in self.feeds],
            "total_kg": round(self.total_kg(), 2),
            "conversion_pct": round(self.conversion * 100.0, 1),
            "biogas_m3": round(self.biogas_m3(), 4),
            "methane_m3": round(self.methane_m3(), 4),
            "ch4_frac": round(self.ch4_fraction(), 3),
            "rate_m3_day": round(self.rate_m3_day, 4),
            "peak_rate_m3_day": round(self.peak_rate_m3_day, 4),
            "energy_kwh": round(kwh, 3),
            "value": round(kwh * energy_price, 2),
            "currency": currency,
            "co2e_kg": round(self.co2e_avoided_kg(), 2),
            "activity": round(activity_index(temp_c), 1),
            "days_to_harvest": None if d2h == float("inf") else round(d2h, 2),
            "ready": self.ready_to_harvest(),
            "cn": self.cn_advice(),
            "age_days": round((time.time() - self.started) / 86400.0, 2),
        }

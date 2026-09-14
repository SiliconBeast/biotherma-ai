# BioTherma-AI

Edge-AI monitor for a small-scale anaerobic bio-digester, built on the Arduino
UNO Q.

## What it does

The STM32 microcontroller polls an AM2320 in the reactor headspace and pushes
temperature and humidity to the Qualcomm Linux side over Bridge. Two things
then run on Linux:

1. **Anomaly detection.** A GRU forecaster learns the chamber's normal
   thermal and humidity dynamics during stable operation and scores the
   one-step forecast residual. A sustained residual means the reactor has
   stopped behaving the way it did during training — heater fault, lid seal,
   stalled culture.

2. **A process model.** `python/digester.py` estimates biogas output, harvest
   timing, carbon-to-nitrogen balance and energy value from the operator's
   feed log and the measured temperature, using published yield coefficients
   and first-order hydrolysis kinetics with a modified-Arrhenius temperature
   correction.

There is no gas sensor in the build. Nothing here claims to measure methane;
the yield figures are modelled from feed mass and measured temperature, and
the dashboard says so.

Status appears on the UNO Q's two microcontroller-side RGB LEDs. LED3 shows
what the reactor is doing, LED4 shows whether the readings can be trusted.

## Bill of materials

- Arduino UNO Q
- AM2320 temperature and humidity sensor

That is the whole list. The data-line pull-up is the STM32 internal, enabled in
firmware. Indicators are on the board. Three wires.

## Wiring

The AM2320 ships in two forms and the one you have decides the wiring.

**3-pin module (single-wire, the default build):**

| AM2320 | UNO Q |
| ------ | ----- |
| VCC    | 3V3   |
| DAT    | D2    |
| GND    | GND   |

**4-pin module (I2C).** Set `SENSOR_SINGLE_WIRE` to 0 in the sketch, then:

| AM2320 | UNO Q |
| ------ | ----- |
| VDD    | 3V3   |
| SDA    | A4 / SDA |
| SCL    | A5 / SCL |
| GND    | GND   |

A 3-pin module cannot speak I2C — that needs two data lines.

Every rail is 3.3 V. The UNO Q headers are **not** 5 V like an UNO R3 —
absolute maximum at the pins is 3.6 V.

See `docs/biotherma_schematic.png`.

## Layout

```
app.yaml          app descriptor
sketch/           STM32 firmware
python/           Linux application and process model
assets/           local dashboard served by the web_ui brick
model/            offline training, verification and a bench simulator
cad/              printable enclosure
docs/             schematic and its generator
```

## Getting a model

The app runs without one. With no `model/weights.npz` it stays in
baseline-capture mode: it logs to SQLite and drives the LEDs from the
mesophilic band check alone. That is how you collect training data.

```
# after a stable run of an hour or more
sqlite3 ~/biotherma.db -header -csv \
  "SELECT ts,air_c,rh,probe_c FROM samples ORDER BY ts" > baseline.csv

# on a laptop, not the board
python model/train.py --csv baseline.csv --features air_c,rh
python model/verify.py weights.npz
```

Copy `weights.npz` into `model/` and restart the app. Set `Z_WARN` and
`Z_ALERT` in `python/main.py` from the residual sigmas the trainer prints.

## Developing without hardware

`model/sim.py` drives the whole pipeline with no board attached, with three
injectable faults. Its output is simulated and is for development only.

```
python model/sim.py --realtime
python model/sim.py --fault heater --at 1800
```

## Optional thermistor

The build has no analog channel: a divider needs a physical resistor to
ground and there is no software substitute. If you fit a 10 kΩ NTC and a
10 kΩ resistor later, set `HAS_NTC 1` in the sketch and add `probe_c` to
`BIOTHERMA_FEATURES`. Both paths work.

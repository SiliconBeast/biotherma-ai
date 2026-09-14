#!/usr/bin/env python3
"""Emit the BioTherma-AI schematic as a standalone SVG.

Plain 2D schematic with conventional symbols. Regenerate after any wiring
change so the diagram and the firmware can't drift apart.

    python make_schematic.py > biotherma_schematic.svg
"""

import sys

W, H = 1480, 860
INK = "#12232b"
WIRE = "#12232b"
RAIL_POS = "#b4452f"
RAIL_GND = "#3c4a52"
NOTE = "#6b7c85"
ACCENT = "#2f6b4f"

out = []


def add(s):
    out.append(s)


def text(x, y, s, size=13, anchor="start", fill=INK, weight="400", style=""):
    add(f'<text x="{x}" y="{y}" font-family="IBM Plex Sans, Helvetica, Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}" '
        f'{"font-style=\"italic\"" if style == "i" else ""}>{s}</text>')


def wire(pts, color=WIRE, width=1.6):
    d = " ".join(("M" if i == 0 else "L") + f"{x} {y}" for i, (x, y) in enumerate(pts))
    add(f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="none" '
        f'stroke-linecap="square" stroke-linejoin="miter"/>')


def junction(x, y, color=WIRE):
    add(f'<circle cx="{x}" cy="{y}" r="3.4" fill="{color}"/>')


def resistor(x, y, label, value, vertical=False, flip_label=False):
    """IEC-style box resistor. (x,y) is the top/left lead start."""
    body, lead = 34, 13
    if vertical:
        add(f'<rect x="{x - 7}" y="{y + lead}" width="14" height="{body}" '
            f'fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>')
        wire([(x, y), (x, y + lead)])
        wire([(x, y + lead + body), (x, y + lead * 2 + body)])
        tx = x - 13 if flip_label else x + 13
        anch = "end" if flip_label else "start"
        text(tx, y + lead + 14, label, 12.5, anch, weight="600")
        text(tx, y + lead + 28, value, 12.5, anch, fill=NOTE)
        return y + lead * 2 + body
    add(f'<rect x="{x + lead}" y="{y - 7}" width="{body}" height="14" '
        f'fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>')
    wire([(x, y), (x + lead, y)])
    wire([(x + lead + body, y), (x + lead * 2 + body, y)])
    ty = y + 26 if flip_label else y - 14
    text(x + lead + body / 2, ty, label, 12.5, "middle", weight="600")
    text(x + lead + body / 2, ty + (14 if flip_label else -13), value, 12.5, "middle", fill=NOTE)
    return x + lead * 2 + body


def thermistor(x, y, label, value):
    """Resistor box with the diagonal stroke that marks a thermistor."""
    body, lead = 34, 13
    add(f'<rect x="{x - 7}" y="{y + lead}" width="14" height="{body}" '
        f'fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>')
    add(f'<path d="M{x - 15} {y + lead + body + 6} L{x + 15} {y + lead - 6}" '
        f'stroke="{INK}" stroke-width="1.6" fill="none"/>')
    add(f'<path d="M{x + 8} {y + lead - 6} L{x + 15} {y + lead - 6} L{x + 15} {y + lead + 1}" '
        f'stroke="{INK}" stroke-width="1.6" fill="none"/>')
    wire([(x, y), (x, y + lead)])
    wire([(x, y + lead + body), (x, y + lead * 2 + body)])
    text(x + 24, y + lead + 14, label, 12.5, "start", weight="600")
    text(x + 24, y + lead + 28, value, 12.5, "start", fill=NOTE)
    return y + lead * 2 + body


def led(x, y, label, colour):
    """LED pointing down: anode at top, cathode at bottom."""
    size = 13
    add(f'<path d="M{x - size} {y} L{x + size} {y} L{x} {y + 22} Z" '
        f'fill="{colour}" stroke="{INK}" stroke-width="1.6" stroke-linejoin="round"/>')
    wire([(x - size - 2, y + 22), (x + size + 2, y + 22)], width=1.8)
    for dx, dy in ((6, -6), (12, 0)):
        add(f'<path d="M{x + 16 + dx} {y + 2 + dy} l9 -9 M{x + 22 + dx} {y - 7 + dy} '
            f'l3 0 l0 3" stroke="{INK}" stroke-width="1.3" fill="none"/>')
    text(x, y - 10, label, 12.5, "middle", weight="600")


def pnp(x, y):
    """PNP transistor, emitter up. Base enters from the left."""
    r = 26
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>')
    wire([(x - 26, y), (x - 9, y)])                       # base lead
    wire([(x - 9, y - 15), (x - 9, y + 15)], width=2.6)   # base bar
    wire([(x - 9, y - 9), (x + 13, y - 22)])              # to emitter
    wire([(x + 13, y - 22), (x + 13, y - 34)])
    wire([(x - 9, y + 9), (x + 13, y + 22)])              # to collector
    wire([(x + 13, y + 22), (x + 13, y + 34)])
    # Emitter arrow points into the base -- this is what makes it PNP.
    add(f'<path d="M{x - 3} {y - 12} l12 -7 l-2 9 Z" fill="{INK}" stroke="{INK}"/>')
    text(x + 34, y - 26, "E", 12, "start", fill=NOTE)
    text(x + 34, y + 32, "C", 12, "start", fill=NOTE)
    text(x - 34, y - 8, "B", 12, "end", fill=NOTE)


def gnd(x, y):
    wire([(x, y), (x, y + 12)], RAIL_GND)
    for i, w in enumerate((18, 11, 5)):
        yy = y + 12 + i * 5
        wire([(x - w / 2, yy), (x + w / 2, yy)], RAIL_GND, 2.0)


def rail_tick(x, y, label):
    wire([(x, y), (x, y - 12)], RAIL_POS)
    wire([(x - 11, y - 12), (x + 11, y - 12)], RAIL_POS, 2.4)
    text(x, y - 19, label, 12, "middle", fill=RAIL_POS, weight="600")


# --------------------------------------------------------------------------
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add(f'<rect width="{W}" height="{H}" fill="#fbfaf7"/>')

text(48, 58, "BioTherma-AI", 24, weight="600")
text(48, 82, "Bio-digester monitoring shield for Arduino UNO Q", 14, fill=NOTE)
add(f'<path d="M48 100 L{W - 48} 100" stroke="{INK}" stroke-width="1.2" opacity="0.25"/>')
text(W - 48, 58, "Rev C", 13, "end", fill=NOTE)
text(W - 48, 78, "All logic 3.3 V", 13, "end", fill=RAIL_POS, weight="600")

# ---------------------------------------------------------------- UNO Q block
bx, by, bw, bh = 80, 180, 272, 460
add(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="4" fill="#ffffff" '
    f'stroke="{INK}" stroke-width="2"/>')
text(bx + bw / 2, by + 60, "Arduino UNO Q", 17, "middle", weight="600")
text(bx + bw / 2, by + 82, "STM32U585 + QRB2210", 12.5, "middle", fill=NOTE)
text(bx + bw / 2, by + 102, "headers are 3.3 V logic", 12.5, "middle", fill=RAIL_POS)

PINS = [("3V3", 30), ("A4 / SDA", 150), ("A5 / SCL", 190), ("GND", 420)]
pin_y = {}
for name, dy in PINS:
    y = by + dy
    pin_y[name] = y
    wire([(bx + bw, y), (bx + bw + 22, y)])
    text(bx + bw - 12, y + 4, name, 13, "end", weight="600")

X_EDGE = bx + bw + 22
X_END = 1400

# ---------------------------------------------------------------- rails
Y_3V3 = pin_y["3V3"]
Y_GBUS = 700
wire([(X_EDGE, Y_3V3), (X_END, Y_3V3)], RAIL_POS, 2.2)
text(X_END + 6, Y_3V3 + 4, "3V3", 13, "start", fill=RAIL_POS, weight="600")

wire([(X_EDGE, pin_y["GND"]), (390, pin_y["GND"]), (390, Y_GBUS), (X_END, Y_GBUS)],
     RAIL_GND, 2.2)
text(X_END + 6, Y_GBUS + 4, "GND", 13, "start", fill=RAIL_GND, weight="600")

# ---------------------------------------------------------------- AM2320
ax, ay = 480, 264
aw, ah = 168, 140
add(f'<rect x="{ax}" y="{ay}" width="{aw}" height="{ah}" rx="3" fill="#ffffff" '
    f'stroke="{INK}" stroke-width="1.8"/>')
text(ax + aw / 2, ay + 28, "AM2320", 15, "middle", weight="600")
text(ax + aw / 2, ay + ah + 22, "temperature + humidity, I\u00b2C 0x5C", 12, "middle", fill=NOTE)

am_pins = {"VDD": ay + 54, "SDA": ay + 84, "SCL": ay + 110, "GND": ay + 132}
for nm, y in am_pins.items():
    wire([(ax, y), (ax - 24, y)])
    text(ax + 12, y + 4, nm, 12, "start", fill=NOTE)

wire([(ax - 24, am_pins["VDD"]), (ax - 24, Y_3V3)], RAIL_POS)
junction(ax - 24, Y_3V3, RAIL_POS)

wire([(ax - 24, am_pins["GND"]), (ax - 58, am_pins["GND"]), (ax - 58, Y_GBUS)], RAIL_GND)
junction(ax - 58, Y_GBUS, RAIL_GND)

X_SDA, X_SCL = 452, 414
wire([(ax - 24, am_pins["SDA"]), (X_SDA, am_pins["SDA"]), (X_SDA, pin_y["A4 / SDA"]),
      (X_EDGE, pin_y["A4 / SDA"])])
wire([(ax - 24, am_pins["SCL"]), (X_SCL, am_pins["SCL"]), (X_SCL, pin_y["A5 / SCL"]),
      (X_EDGE, pin_y["A5 / SCL"])])

text(392, 156, "Bus pull-ups are the STM32 internal pull-ups,", 12.5, "start",
     fill=RAIL_POS, weight="600")
text(392, 172, "enabled in firmware. Roughly 40 k\u03a9 \u2014 weak, but enough for one", 12.5,
     "start", fill=NOTE)
text(392, 188, "device on a short bus at 100 kHz.", 12.5, "start", fill=NOTE)

text(452, 472, "Four wires and one sensor.", 13, "start", weight="600")
text(452, 492, "The AM2320 reports both the chamber temperature and the", 12.5, "start", fill=NOTE)
text(452, 508, "headspace humidity, so no analog channel and no divider are", 12.5, "start", fill=NOTE)
text(452, 524, "needed. Mount it inside the headspace, not outside.", 12.5, "start", fill=NOTE)

# ------------------------------------------------- on-board status indicators
px, py = 1010, 300
add(f'<rect x="{px}" y="{py}" width="330" height="176" rx="4" fill="#ffffff" '
    f'stroke="{INK}" stroke-width="1.8" stroke-dasharray="6 4"/>')
text(px + 165, py + 32, "On-board status LEDs", 15, "middle", weight="600")
text(px + 165, py + 54, "no external parts, nothing to wire", 12, "middle", fill=NOTE)

rows = [
    ("LED3", "process state", "#4f9d5d"),
    ("LED4", "sensor health", "#c1453a"),
]
for i, (nm, desc, col) in enumerate(rows):
    y = py + 92 + i * 34
    add(f'<circle cx="{px + 36}" cy="{y - 4}" r="8" fill="{col}" stroke="{INK}" '
        f'stroke-width="1.4"/>')
    text(px + 56, y, nm, 13, "start", weight="600")
    text(px + 112, y, desc, 12.5, "start", fill=NOTE)

text(px + 165, py + 162, "RGB, MCU-side, active LOW", 12, "middle", fill=ACCENT,
     weight="600")

text(px + 165, py + 210, "The UNO Q carries four on-board RGB LEDs.", 12.5, "middle", fill=NOTE)
text(px + 165, py + 226, "Two are driven by the Linux MPU, two by the STM32.", 12.5, "middle", fill=NOTE)
text(px + 165, py + 242, "This design uses the two MCU-side LEDs, so there are", 12.5, "middle", fill=NOTE)
text(px + 165, py + 258, "no indicator LEDs, resistors or switching transistor", 12.5, "middle", fill=NOTE)
text(px + 165, py + 274, "in the bill of materials.", 12.5, "middle", fill=NOTE)

# ---------------------------------------------------------------- notes
add(f'<path d="M48 {H - 108} L{W - 48} {H - 108}" stroke="{INK}" stroke-width="1.2" opacity="0.25"/>')
notes = [
    ("Every rail is 3.3 V. The UNO Q headers are not 5 V like an UNO R3.", True),
    ("A 5 V divider or pull-up exceeds the 3.6 V absolute maximum on the analog and digital pins.", False),
    ("Status indication uses the two MCU-side on-board RGB LEDs (LED3 and LED4), so no GPIO is spent on it.", True),
    ("One component total: the AM2320. If I2C reads fail CRC intermittently, fit real 4.7k-10k pull-ups.", False),
]
for i, (n, strong) in enumerate(notes):
    text(48, H - 82 + i * 19, n, 12.5, fill=INK if strong else NOTE,
         weight="600" if strong else "400")

add('</svg>')

sys.stdout.write("\n".join(out) + "\n")

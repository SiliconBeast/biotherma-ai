#!/usr/bin/env python3
"""Build the BioTherma-AI enclosure and export printable STLs.

Two parts: a wall-mountable base that the UNO Q bolts into, and a lid with a
viewing window over the board's own status LEDs. Sized for the standard Arduino
UNO footprint, which the UNO Q shares. No external indicator hardware.

    pip install trimesh manifold3d
    python build_enclosure.py

Writes enclosure_base.stl and enclosure_lid.stl next to this file.

Print notes: PETG rather than PLA. The enclosure sits next to a warm reactor
and PLA starts creeping around 55 C. 0.2 mm layers, 3 perimeters, 20% infill.
The base prints cavity-up with no supports; the lid prints face-down.
"""

import os

import numpy as np
import trimesh
from trimesh.creation import box, cylinder

# --------------------------------------------------------------------------
# Parameters -- everything downstream derives from these
# --------------------------------------------------------------------------

WALL = 2.4               # side wall thickness
FLOOR = 2.6              # base floor thickness
LID_T = 3.0              # lid plate thickness

BOARD_L, BOARD_W = 68.58, 53.34     # Arduino UNO footprint
BOARD_CLEAR = 4.3                   # gap between board edge and inner wall
STANDOFF_H = 6.0                    # lifts the board off the floor
STANDOFF_D = 6.4
STANDOFF_PILOT = 2.5                # self-tapping for M3

# Arduino UNO mounting holes, measured from the board's lower-left corner.
UNO_HOLES = [(13.97, 2.54), (15.24, 50.8), (66.04, 35.56), (66.04, 7.62)]

INNER_L = BOARD_L + 2 * BOARD_CLEAR
INNER_W = BOARD_W + 2 * BOARD_CLEAR
INNER_H = 28.0                      # clearance above the floor

OUTER_L = INNER_L + 2 * WALL
OUTER_W = INNER_W + 2 * WALL
OUTER_H = INNER_H + FLOOR

# Board origin inside the cavity
BX0 = WALL + BOARD_CLEAR
BY0 = WALL + BOARD_CLEAR
BOARD_Z = FLOOR + STANDOFF_H        # top face of the standoffs

USB_W, USB_H = 13.0, 8.0            # USB-C cutout, generous for a plug shell
USB_Z = BOARD_Z + 1.0               # sits just above the PCB

GLAND_D = 7.0                       # sensor cable pass-throughs
# Status indication uses the UNO Q's own MCU-side RGB LEDs, so the lid needs a
# viewing window rather than LED holes. The slot is deliberately generous:
# measure where LED3 and LED4 actually sit on your board and adjust WINDOW_X /
# WINDOW_Y before printing.
WINDOW_L, WINDOW_W = 34.0, 10.0
WINDOW_X = 0.0                      # offset from lid centre, +X toward USB end
WINDOW_Y = 15.0

BOSS_D, BOSS_PILOT = 7.0, 2.5       # lid screw bosses in the base corners
BOSS_INSET = 6.2
LID_SCREW_D = 3.4

TAB_L, TAB_W, TAB_T = 16.0, 22.0, 3.4   # wall-mount ears
TAB_HOLE_D = 4.5

VENT_W, VENT_H = 2.6, 12.0
VENT_COUNT = 5
VENT_PITCH = 6.0

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def bx(size, centre):
    """Axis-aligned box by size and centre point."""
    T = np.eye(4)
    T[:3, 3] = centre
    return box(extents=size, transform=T)


def cyl(d, h, centre, axis="z"):
    T = np.eye(4)
    if axis == "x":
        T[:3, :3] = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3]
    elif axis == "y":
        T[:3, :3] = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3]
    T[:3, 3] = centre
    return cylinder(radius=d / 2.0, height=h, sections=48, transform=T)


def vent_bank(x_centre, y, z_centre, along="x"):
    """A row of slots cut through a wall."""
    cuts = []
    span = (VENT_COUNT - 1) * VENT_PITCH
    for i in range(VENT_COUNT):
        off = -span / 2 + i * VENT_PITCH
        if along == "x":
            cuts.append(bx((VENT_W, WALL * 4, VENT_H), (x_centre + off, y, z_centre)))
        else:
            cuts.append(bx((WALL * 4, VENT_W, VENT_H), (x_centre, y + off, z_centre)))
    return cuts


def boss_positions():
    return [(BOSS_INSET, BOSS_INSET),
            (OUTER_L - BOSS_INSET, BOSS_INSET),
            (BOSS_INSET, OUTER_W - BOSS_INSET),
            (OUTER_L - BOSS_INSET, OUTER_W - BOSS_INSET)]


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------

def build_base():
    shell = bx((OUTER_L, OUTER_W, OUTER_H), (OUTER_L / 2, OUTER_W / 2, OUTER_H / 2))

    cavity = bx((INNER_L, INNER_W, INNER_H + 1),
                (OUTER_L / 2, OUTER_W / 2, FLOOR + (INNER_H + 1) / 2))
    part = shell.difference(cavity)

    # Wall-mount ears, one each side, with the fixing holes clear of the body.
    ears = []
    for sign, y in ((-1, -TAB_L / 2), (1, OUTER_W + TAB_L / 2)):
        ear = bx((TAB_W, TAB_L, TAB_T), (OUTER_L / 2, y, TAB_T / 2))
        ears.append(ear)
    part = trimesh.boolean.union([part] + ears)

    additions = []
    # Board standoffs
    for hx, hy in UNO_HOLES:
        additions.append(cyl(STANDOFF_D, STANDOFF_H,
                             (BX0 + hx, BY0 + hy, FLOOR + STANDOFF_H / 2)))
    # Lid screw bosses, full cavity height
    for bx_, by_ in boss_positions():
        additions.append(cyl(BOSS_D, INNER_H, (bx_, by_, FLOOR + INNER_H / 2)))
    part = trimesh.boolean.union([part] + additions)

    cuts = []
    # Standoff pilot holes
    for hx, hy in UNO_HOLES:
        cuts.append(cyl(STANDOFF_PILOT, STANDOFF_H + 4,
                        (BX0 + hx, BY0 + hy, FLOOR + STANDOFF_H / 2 + 1)))
    # Boss pilot holes
    for bx_, by_ in boss_positions():
        cuts.append(cyl(BOSS_PILOT, INNER_H, (bx_, by_, FLOOR + INNER_H / 2 + 2)))
    # Wall-mount holes
    for y in (-TAB_L / 2, OUTER_W + TAB_L / 2):
        cuts.append(cyl(TAB_HOLE_D, TAB_T + 4, (OUTER_L / 2, y, TAB_T / 2)))

    # USB-C cutout on the -X wall, at the board's USB end.
    cuts.append(bx((WALL * 4, USB_W, USB_H),
                   (0, OUTER_W / 2, USB_Z + USB_H / 2)))

    # Sensor cable pass-throughs on the +X wall: one for the slurry probe,
    # one for the headspace sensor. Spaced so two glands do not foul.
    for dy in (-11.0, 11.0):
        cuts.append(cyl(GLAND_D, WALL * 4,
                        (OUTER_L, OUTER_W / 2 + dy, FLOOR + 11.0), axis="x"))

    # Convection slots low on both long walls.
    cuts += vent_bank(OUTER_L / 2, 0, FLOOR + INNER_H / 2, along="x")
    cuts += vent_bank(OUTER_L / 2, OUTER_W, FLOOR + INNER_H / 2, along="x")

    part = part.difference(trimesh.boolean.union(cuts))
    return part


# --------------------------------------------------------------------------
# Lid
# --------------------------------------------------------------------------

def build_lid():
    plate = bx((OUTER_L, OUTER_W, LID_T), (OUTER_L / 2, OUTER_W / 2, LID_T / 2))

    # Locating lip that drops into the cavity.
    lip_out = bx((INNER_L - 0.4, INNER_W - 0.4, 2.0),
                 (OUTER_L / 2, OUTER_W / 2, LID_T + 1.0))
    lip_in = bx((INNER_L - 0.4 - 2 * 1.6, INNER_W - 0.4 - 2 * 1.6, 2.4),
                (OUTER_L / 2, OUTER_W / 2, LID_T + 1.0))
    lip = lip_out.difference(lip_in)
    part = trimesh.boolean.union([plate, lip])

    cuts = []
    # Viewing window over the on-board status LEDs.
    cuts.append(bx((WINDOW_L, WINDOW_W, LID_T + 6),
                   (OUTER_L / 2 + WINDOW_X, OUTER_W / 2 + WINDOW_Y, LID_T / 2)))
    # Lid screws
    for bx_, by_ in boss_positions():
        cuts.append(cyl(LID_SCREW_D, LID_T + 6, (bx_, by_, LID_T / 2)))
    # Vent slots, away from the LED row.
    for i in range(VENT_COUNT):
        off = -(VENT_COUNT - 1) * VENT_PITCH / 2 + i * VENT_PITCH
        cuts.append(bx((VENT_W, VENT_H, LID_T + 6),
                       (OUTER_L / 2 + off, OUTER_W / 2 - 15.0, LID_T / 2)))

    part = part.difference(trimesh.boolean.union(cuts))
    return part


# --------------------------------------------------------------------------

def main():
    for name, mesh in (("enclosure_base", build_base()), ("enclosure_lid", build_lid())):
        path = os.path.join(HERE, name + ".stl")
        mesh.export(path)
        print(f"{name}.stl  watertight={mesh.is_watertight}  "
              f"volume={mesh.volume / 1000:.1f} cm3  "
              f"bbox={np.round(mesh.extents, 1)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Generate English PNG figures with pure Pillow (matplotlib native renderer is
blocked by this machine's app-control policy; PIL is allowed).

All plotted numbers are measured on this machine (2026-09-28); no invented data.
Output: docs/figures/fig1..fig4 (drawn at 2x, downscaled for anti-aliasing).
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

S = 2  # supersampling factor

INK = "#26221C"
SUB = "#444441"
MUT = "#5F5E5A"
BLUE, BLUE_L = "#185FA5", "#E6F1FB"
TEAL, TEAL_L = "#0F6E56", "#E1F5EE"
CORAL, CORAL_L = "#993C1D", "#FAECE7"
AMBER, AMBER_L = "#854F0B", "#FAEEDA"
GREEN, GREEN_L = "#3B6D11", "#EAF3DE"
PURPLE, PURPLE_L = "#534AB7", "#EEEDFE"
RED = "#A32D2D"
GRID = "#D8D5CC"

FONT_DIR = "C:/Windows/Fonts"
def F(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size * S)


def canvas(w, h):
    img = Image.new("RGB", (w * S, h * S), "white")
    return img, ImageDraw.Draw(img)


def text_c(d, cx, cy, s, font, fill=INK):
    bb = d.textbbox((0, 0), s, font=font)
    d.text((cx - (bb[2] - bb[0]) / 2, cy - (bb[3] - bb[1]) / 2 - bb[1]),
           s, font=font, fill=fill)


def box(d, x, y, w, h, title, sub, fill, edge, title_size=15, sub_size=12):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S],
                        radius=10 * S, fill=fill, outline=edge, width=2 * S)
    lines_t = title.split("\n")
    lines_s = sub.split("\n") if sub else []
    total = len(lines_t) * (title_size + 6) + len(lines_s) * (sub_size + 6)
    cy = y + h / 2 - (total / 2) / S * S / S  # anchor middle
    yy = y + (h - (len(lines_t) * (title_size + 7) + len(lines_s) * (sub_size + 7)) / 1.0) / 2
    for ln in lines_t:
        text_c(d, (x + w / 2) * S, yy * S + title_size * S / 2, ln,
               F(title_size, bold=True))
        yy += title_size + 7
    for ln in lines_s:
        text_c(d, (x + w / 2) * S, yy * S + sub_size * S / 2, ln, F(sub_size), fill=SUB)
        yy += sub_size + 7


def arrow(d, x1, y1, x2, y2, color=MUT, width=3, head=12):
    import math
    d.line([x1 * S, y1 * S, x2 * S, y2 * S], fill=color, width=width * S)
    ang = math.atan2(y2 - y1, x2 - x1)
    h = head * S
    p1 = (x2 * S, y2 * S)
    p2 = (x2 * S - h * math.cos(ang - 0.42), y2 * S - h * math.sin(ang - 0.42))
    p3 = (x2 * S - h * math.cos(ang + 0.42), y2 * S - h * math.sin(ang + 0.42))
    d.polygon([p1, p2, p3], fill=color)


def save(img, name, w, h):
    img = img.resize((w, h), Image.LANCZOS)
    img.save(os.path.join(OUT, name))
    print(" -", name)


# ---------------------------------------------------------------- Figure 1
W, H = 2040, 880
img, d = canvas(W, H)
text_c(d, W * S / 2, 30 * S, "LeLamp-FlyBrain Phase-1 System Architecture",
       F(22, bold=True))

y0, hh = 370, 310
box(d, 50, y0, 410, hh, "L0 Scenario Layer",
    "DeepSeek -> Qwen -> Mock\nfailover chain\nJSON event stream + gamma",
    BLUE_L, BLUE)
box(d, 570, y0, 460, hh, "L1 FlyBrain SNN",
    "500-node fixed sparse graph\n10 DN readout: dq = W * s_DN\ngamma modulation (Fig. 2)",
    PURPLE_L, PURPLE)
box(d, 1140, y0, 370, hh, "L2 Safety Layer",
    "rate limit 0.06 rad/step\njoint-limit clamp\ncollision monitor",
    TEAL_L, TEAL)
box(d, 1620, y0, 370, hh, "L3 Physics + Body",
    "MuJoCo 240 Hz\nkp=17.8 STS3215 servos\n(PyBullet fallback)",
    CORAL_L, CORAL)

arrow(d, 460, y0 + hh / 2, 570, y0 + hh / 2, color=BLUE)
arrow(d, 1030, y0 + hh / 2, 1140, y0 + hh / 2, color=PURPLE)
arrow(d, 1510, y0 + hh / 2, 1620, y0 + hh / 2, color=TEAL)

# feedback loop
arrow(d, 1805, y0 + hh, 800, 800, color=AMBER, width=3)
arrow(d, 800, 800, 800, y0 + hh, color=AMBER, width=3)
text_c(d, 1310 * S, 795 * S,
       "closed-loop proprioception: ON/OFF joint-error sensory channels (50 Hz)",
       F(13), fill=AMBER)
text_c(d, W * S / 2, 855 * S,
       "SNN 100 Hz internal steps  |  control 50 Hz  |  physics 240 Hz  |  zero dataset, pure simulation",
       F(12), fill=MUT)
save(img, "fig1_architecture.png", W // S * 2 // 2, H // S * 2 // 2)

# ---------------------------------------------------------------- Figure 2
W, H = 1920, 1040
img, d = canvas(W, H)
text_c(d, W * S / 2, 34 * S,
       "Neuromodulation Vector gamma = [DA, OA, 5-HT]: Three Pathways",
       F(22, bold=True))

box(d, 70, 420, 400, 300, "LLM event stream",
    "each event carries\ngamma in [0,1]^3", BLUE_L, BLUE)

box(d, 630, 150, 580, 150, "5-HT -> leak time constant", "", CORAL_L, CORAL)
text_c(d, 920 * S, 262 * S, "leak = clip(0.70 + 0.10*(5HT - 0.5))",
       F(13), fill=SUB)
box(d, 630, 460, 580, 150, "OA -> sensory injection gain", "", AMBER_L, AMBER)
text_c(d, 920 * S, 572 * S, "gain = 0.5 + 1.5 * OA   (arousal)",
       F(13), fill=SUB)
box(d, 630, 770, 580, 150, "DA -> three-factor learning gate", "", GREEN_L, GREEN)
text_c(d, 920 * S, 882 * S, "W += eta * (0.2 + 0.8*DA) * TD_error * E",
       F(13), fill=SUB)

box(d, 1370, 390, 480, 380, "FlyBrain SNN + DN readout",
    "h <- tanh(leak*h + msg + x + b)\n500 nodes, fixed edges\ndq = W * s_DN",
    PURPLE_L, PURPLE)

arrow(d, 470, 520, 630, 225, color=CORAL)
arrow(d, 470, 560, 630, 535, color=AMBER)
arrow(d, 470, 620, 630, 845, color=GREEN)
arrow(d, 1210, 225, 1370, 500, color=CORAL)
arrow(d, 1210, 535, 1370, 580, color=AMBER)
arrow(d, 1210, 845, 1370, 660, color=GREEN)

text_c(d, W * S / 2, 950 * S,
       "gamma changes HOW the same fixed topology runs (gain / time constant / learning gate), never its wiring.",
       F(13), fill=MUT)
text_c(d, W * S / 2, 990 * S,
       "Engineering analogy, not a biological equivalence claim (FlyDoom convention).",
       F(12), fill=MUT)
text_c(d, W * S / 2, 1022 * S,
       "Known limitation: current 5-HT path modulates leak only within [0.70, 0.75] - nearly decorative; Phase-2 fix planned.",
       F(12), fill=RED)
save(img, "fig2_gamma_pathways.png", W // 2, H // 2)

# ---------------------------------------------------------------- Figure 3
W, H = 2040, 800
img, d = canvas(W, H)
text_c(d, W * S / 2, 34 * S, "FlyDoom Three Pillars as Implemented in This Project",
       F(22, bold=True))

bw, bh, y0 = 610, 470, 170
box(d, 60, y0, bw, bh, "Pillar 1\nFixed Sparse Topology",
    "500-node COO mock connectome\nedges frozen after seed\nlog1p(synapse) init, subcritical\nhomeostatic bias",
    BLUE_L, BLUE)
box(d, 720, y0, bw, bh, "Pillar 2\nDN-Only Readout",
    "10 descending neurons (DN)\ndq = W * s_DN,  W in R^(5x10)\nridge-calibrated (no backprop)\nmean tracking err 0.040 rad",
    TEAL_L, TEAL)
box(d, 1380, y0, bw, bh, "Pillar 3\nThree-Factor Learning",
    "eligibility trace E x DA-gated TD\nno backpropagation\nstatus: unstable in T1 (honest\nnegative result, see report)",
    CORAL_L, CORAL)

text_c(d, W * S / 2, 720 * S,
       "No backpropagation  |  No learned connectivity  |  No dataset  |  Ablation-ready (ER / degree-preserving rewiring null models)",
       F(13), fill=MUT)
save(img, "fig3_three_pillars.png", W // 2, H // 2)

# ---------------------------------------------------------------- Figure 4
W, H = 2120, 840
img, d = canvas(W, H)
text_c(d, W * S / 2, 34 * S, "Measured Results on This Machine (2026-09-28)",
       F(22, bold=True))


def bar_panel(x0, y0, pw, ph, title):
    text_c(d, (x0 + pw / 2) * S, y0 * S, title, F(15, bold=True))
    return x0, y0 + 50, pw, ph - 50


states = ["Emotional\nComfort", "Fear\nArousal", "Approach\nReward",
          "Vigilance\nDefense", "Exploration\nSeeking", "Grooming\nRest"]
errs = [0.060, 0.022, 0.046, 0.024, 0.026, 0.061]

px, py, pw, ph = bar_panel(150, 90, 930, 700,
                           "(a) T1 per-state tracking error, ridge-calibrated W")
top = py + 40
bot = py + ph - 120
ymax = 0.60


def ymap(v):
    return bot - (v / ymax) * (bot - top)


# grid + y ticks
for v in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    yy = ymap(v)
    d.line([px * S, yy * S, (px + pw) * S, yy * S], fill=GRID, width=S)
    d.text(((px - 70) * S, (yy - 12) * S), f"{v:.1f}", font=F(12), fill=MUT)
d.line([px * S, top * S, px * S, bot * S], fill=MUT, width=2 * S)
d.line([px * S, bot * S, (px + pw) * S, bot * S], fill=MUT, width=2 * S)

n = 6
slot = pw / n
bwid = slot * 0.55
for i, e in enumerate(errs):
    cx = px + slot * (i + 0.5)
    d.rectangle([(cx - bwid / 2) * S, ymap(e) * S, (cx + bwid / 2) * S, bot * S],
                fill=BLUE)
    text_c(d, cx * S, (ymap(e) - 22) * S, f"{e:.3f}", F(11), fill=INK)
    for j, ln in enumerate(states[i].split("\n")):
        text_c(d, cx * S, (bot + 30 + j * 24) * S, ln, F(11), fill=SUB)

for v, col, lbl, dash in [(0.542, RED, "zero-W (no control) = 0.542", True),
                          (0.040, TEAL, "mean = 0.040 rad (~2.3 deg)", False)]:
    yy = ymap(v)
    if dash:
        xcur = px
        while xcur < px + pw:
            d.line([xcur * S, yy * S, min(xcur + 18, px + pw) * S, yy * S],
                   fill=col, width=3 * S)
            xcur += 30
    else:
        d.line([px * S, yy * S, (px + pw) * S, yy * S], fill=col, width=2 * S)
    d.text(((px + 12) * S, (yy - 30) * S), lbl, font=F(12, bold=True), fill=col)
text_c(d, (px + pw / 2) * S, (py + ph - 30) * S,
       "deterministic pose-tracking error (rad), lower is better", F(12), fill=MUT)

# panel b
demo_states = ["Vigilance\nDefense", "Approach\nReward", "Emotional\nComfort"]
sweep = [1.624, 0.754, 0.250]
track = [0.042, 0.048, 0.050]

px2, py2, pw2, ph2 = bar_panel(1210, 90, 860, 700,
                               "(b) Continuous motion under full SNN control (6 s)")
top2 = py2 + 40
bot2 = py2 + ph2 - 120
ymax2 = 2.0


def ymap2(v):
    return bot2 - (v / ymax2) * (bot2 - top2)


for v in [0, 0.5, 1.0, 1.5, 2.0]:
    yy = ymap2(v)
    d.line([px2 * S, yy * S, (px2 + pw2) * S, yy * S], fill=GRID, width=S)
    d.text(((px2 - 60) * S, (yy - 12) * S), f"{v:.1f}", font=F(12), fill=MUT)
d.line([px2 * S, top2 * S, px2 * S, bot2 * S], fill=MUT, width=2 * S)
d.line([px2 * S, bot2 * S, (px2 + pw2) * S, bot2 * S], fill=MUT, width=2 * S)

slot2 = pw2 / 3
bwid2 = slot2 * 0.45
for i, s_ in enumerate(sweep):
    cx = px2 + slot2 * (i + 0.5)
    d.rectangle([(cx - bwid2 / 2) * S, ymap2(s_) * S, (cx + bwid2 / 2) * S, bot2 * S],
                fill=PURPLE)
    text_c(d, cx * S, (ymap2(s_) - 26) * S, f"{s_:.3f} rad", F(12, bold=True),
           fill=INK)
    text_c(d, cx * S, (ymap2(s_) - 58) * S, f"track err {track[i]:.3f}",
           F(11), fill=MUT)
    for j, ln in enumerate(demo_states[i].split("\n")):
        text_c(d, cx * S, (bot2 + 30 + j * 24) * S, ln, F(11), fill=SUB)
text_c(d, (px2 + pw2 / 2) * S, (py2 + ph2 - 30) * S,
       "joint-0 (base yaw) motion range, peak-to-peak (rad)", F(12), fill=MUT)

save(img, "fig4_results.png", W // 2, H // 2)
print("figures written to", OUT)

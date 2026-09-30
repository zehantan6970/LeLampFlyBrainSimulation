# -*- coding: utf-8 -*-
"""Phase-2 Level A 正式运行（72 评估点）收敛分析 + fig6 生成。

纯 numpy 逐元素运算（本机 BLAS GEMM 不可用，禁用手动之外的 matmul/corrcoef），
Pillow 4x 超采样出图。输出：
  - docs/figures/fig6_evolve_levelA.png（三面板）
  - 控制台打印报告所需全部统计量

用法：python tools/analyze_levelA.py
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL = os.path.join(ROOT, "data", "evolve_modulation.jsonl")
OUT = os.path.join(ROOT, "docs", "figures", "fig6_evolve_levelA.png")

DIMS = ["leak_base", "leak_amp", "gain_bias", "gain_slope", "da_floor", "da_slope"]

recs = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
gens = sorted({r["gen"] for r in recs})
fit = np.array([r["fitness"] for r in recs])
err = np.array([r["err_mean"] for r in recs])
auc = np.array([r["auc_gain"] for r in recs])
pen = np.array([r["penalty"] for r in recs])
gidx = np.array([r["gen"] for r in recs])
TH = {k: np.array([r["theta"][k] for r in recs]) for k in DIMS}

# ---- per-generation stats -------------------------------------------------
gen_best = np.array([fit[gidx == g].min() for g in gens])
gen_mean = np.array([fit[gidx == g].mean() for g in gens])

# ---- rank correlations (Spearman, manual; no BLAS) ------------------------
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    num = float((rx * ry).sum())
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return num / den if den > 0 else 0.0

rho = {k: spearman(TH[k], fit) for k in DIMS}

# ---- key subgroups ---------------------------------------------------------
sat = err > 3.0            # 饱和（≈π）
n_sat = int(sat.sum())
best_i = int(np.argmin(fit))
best = recs[best_i]
n_pen = int((pen > 0).sum())

print("== 总体 ==")
print(f"n={len(recs)}  fitness min/median/max = {fit.min():.4f}/{np.median(fit):.4f}/{fit.max():.4f}")
print(f"err=π 饱和个体: {n_sat}/{len(recs)} ({100*n_sat/len(recs):.0f}%)")
print(f"err<3.0 个体: {(~sat).sum()}  其 err 范围 {err[~sat].min():.3f}..{err[~sat].max():.3f}")
print(f"auc_gain 范围 [{auc.min():+.4f}, {auc.max():+.4f}]  mean={auc.mean():+.4f}")
print(f"penalty>0 个体: {n_pen}")
print(f"首代 best {gen_best[0]:.4f} -> 末代 best {gen_best[-1]:.4f}  (全局 best {fit.min():.4f} @ g{best['gen']:02d} i{best['idx']})")
print("== Spearman(θ_dim, fitness)  [负=该维增大有利于降 fitness] ==")
for k in DIMS:
    print(f"  {k:11s} rho={rho[k]:+.3f}")
print("== 最优 θ ==")
print(json.dumps(best["theta"], indent=None))
print(f"最优个体: fit={best['fitness']:.4f} err={best['err_mean']:.3f} auc={best['auc_gain']:+.4f} pen={best['penalty']}")

# 与先验 θ0（默认）对比：g00 全部个体
print(f"== 先验邻域（g00）== best={fit[gidx==0].min():.4f} mean={fit[gidx==0].mean():.4f}")

# ---- fig6: 3 panels --------------------------------------------------------
S = 4
FW, FH = 2200, 900
W, H = FW * S, FH * S
INK, SUB, MUT = "#26221C", "#444441", "#5F5E5A"
BLUE, TEAL, CORAL, PURPLE, RED = "#185FA5", "#0F6E56", "#993C1D", "#534AB7", "#A32D2D"
AMBER, GREEN = "#854F0B", "#3B6D11"
GRID = "#D8D5CC"
FD = "C:/Windows/Fonts"


def f(sz, b=False):
    return ImageFont.truetype(os.path.join(FD, "segoeuib.ttf" if b else "segoeui.ttf"), sz * S)


img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)


def tc(cx, cy, s, ft, fill=INK):
    bb = d.textbbox((0, 0), s, font=ft)
    d.text((cx - (bb[2] - bb[0]) / 2, cy - (bb[3] - bb[1]) / 2 - bb[1]), s, font=ft, fill=fill)


def panel(x0, y0, pw, ph, title):
    tc((x0 + pw / 2) * S, y0 * S, title, f(15, True))
    return x0, y0 + 46, pw, ph - 46


tc(W / 2, 32 * S, "Phase-2 Level A Formal Run: CMA-ES over gamma-mapping coefficients (72 evaluations, 12 gens x 6)", f(20, True))

# (a) fitness per generation -------------------------------------------------
px, py, pw, ph = panel(90, 80, 640, 760, "(a) Fitness per generation (lower = better)")
top, bot = py + 30, py + ph - 110
ymin, ymax = 2.5, 3.6
ym = lambda v: bot - (v - ymin) / (ymax - ymin) * (bot - top)
for v in [2.5, 2.75, 3.0, 3.25, 3.5]:
    d.line([px * S, ym(v) * S, (px + pw) * S, ym(v) * S], fill=GRID, width=S)
    d.text(((px - 70) * S, (ym(v) - 14) * S), f"{v:.2f}", font=f(12), fill=MUT)
xm = lambda g: px + (g + 0.5) / len(gens) * pw
for arr, col, lbl in [(gen_best, TEAL, "best"), (gen_mean, BLUE, "mean")]:
    pts = [(xm(g) * S, ym(v) * S) for g, v in zip(gens, arr)]
    d.line(pts, fill=col, width=3 * S)
    for p in pts:
        d.ellipse([p[0] - 5 * S, p[1] - 5 * S, p[0] + 5 * S, p[1] + 5 * S], fill=col)
    d.text((pts[-1][0] - 60 * S, pts[-1][1] - 30 * S), lbl, font=f(12, True), fill=col)
for g in gens:
    if g % 2 == 0:
        tc(xm(g) * S, (bot + 26) * S, f"g{g:02d}", f(11), fill=SUB)
tc((px + pw / 2) * S, (bot + 60) * S, "CMA-ES generation", f(12), fill=MUT)

# (b) err vs auc scatter ------------------------------------------------------
px, py, pw, ph = panel(830, 80, 640, 760, "(b) No learning signal under any theta")
top, bot = py + 30, py + ph - 110
xmn, xmx = -0.04, 0.015
ymn, ymx = 2.0, 3.3
xm2 = lambda v: px + (v - xmn) / (xmx - xmn) * pw
ym2 = lambda v: bot - (v - ymn) / (ymx - ymn) * (bot - top)
for v in [-0.04, -0.02, 0.0]:
    d.line([xm2(v) * S, top * S, xm2(v) * S, bot * S], fill=GRID, width=S)
    tc(xm2(v) * S, (bot + 26) * S, f"{v:+.2f}", f(11), fill=SUB)
for v in [2.0, 2.5, 3.0]:
    d.line([px * S, ym2(v) * S, (px + pw) * S, ym2(v) * S], fill=GRID, width=S)
    d.text(((px - 55) * S, (ym2(v) - 14) * S), f"{v:.1f}", font=f(12), fill=MUT)
d.line([px * S, ym2(3.1416) * S, (px + pw) * S, ym2(3.1416) * S], fill=RED, width=2 * S)
d.text(((px + 8) * S, (ym2(3.1416) - 30) * S), "err = pi (joint-limit saturation)", font=f(11, True), fill=RED)
gen_cols = [TEAL, BLUE, PURPLE, CORAL]
for r, fv, ev, av, g in zip(recs, fit, err, auc, gidx):
    col = gen_cols[g // 3]
    xx, yy = xm2(av) * S, ym2(min(ev, 3.29)) * S
    d.ellipse([xx - 5 * S, yy - 5 * S, xx + 5 * S, yy + 5 * S], fill=col)
tc((px + pw / 2) * S, (bot + 60) * S, "learning-speed AUC gain (last-10 minus first-10 reward)", f(12), fill=MUT)
tc((px + pw / 2) * S, (top - 16) * S, "color: g00-02 teal, g03-05 blue, g06-08 purple, g09-11 coral", f(10), fill=MUT)

# (c) theta dimension traces (per-gen population mean) ------------------------
px, py, pw, ph = panel(1570, 80, 560, 760, "(c) theta drift (population mean per gen)")
top, bot = py + 30, py + ph - 110
th_mean = {k: np.array([TH[k][gidx == g].mean() for g in gens]) for k in DIMS}
allv = np.concatenate([th_mean[k] for k in DIMS])
ymn3, ymx3 = 0.0, float(allv.max()) * 1.1
ym3 = lambda v: bot - (v - ymn3) / (ymx3 - ymn3) * (bot - top)
xm3 = lambda g: px + (g + 0.5) / len(gens) * pw
dcols = {"leak_base": BLUE, "leak_amp": TEAL, "gain_bias": AMBER,
         "gain_slope": CORAL, "da_floor": PURPLE, "da_slope": GREEN}
for v in [0.0, 0.5, 1.0, 1.5, 2.0]:
    if v <= ymx3:
        d.line([px * S, ym3(v) * S, (px + pw) * S, ym3(v) * S], fill=GRID, width=S)
        d.text(((px - 45) * S, (ym3(v) - 14) * S), f"{v:.1f}", font=f(12), fill=MUT)
for k in DIMS:
    pts = [(xm3(g) * S, ym3(v) * S) for g, v in zip(gens, th_mean[k])]
    d.line(pts, fill=dcols[k], width=2 * S)
    d.text((pts[-1][0] - 95 * S, pts[-1][1] - 8 * S), k, font=f(10, True), fill=dcols[k])
for g in gens:
    if g % 2 == 0:
        tc(xm3(g) * S, (bot + 26) * S, f"g{g:02d}", f(11), fill=SUB)
tc((px + pw / 2) * S, (bot + 60) * S, "CMA-ES generation", f(12), fill=MUT)

# footer
tc(W / 2, (FH - 40) * S,
   "Result: fitness ~ terminal error; AUC ~ 0 under all 72 theta -> inner three-factor loop has no learnable signal; gamma tuning cannot substitute for fixing the learning rule (-> P2-2).",
   f(13), fill=SUB)

img = img.resize((FW, FH), Image.LANCZOS)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
img.save(OUT)
print("saved:", OUT)

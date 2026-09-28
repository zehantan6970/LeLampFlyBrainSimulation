# -*- coding: utf-8 -*-
"""生成 fig5_gamma_symbolic.png：gamma 三通路在 LeLamp 控制回路中的位置（符号版）。

与 fig2（数值版）互补：本图用符号系数 theta=(l0,l1,g0,g1,d0,d1) 表示映射，
这些系数正是 Phase-2 Level A（外环 CMA-ES 进化）的优化目标。
纯 Pillow 绘制（本机 matplotlib FreeType 被策略拦截），2x 超采样抗锯齿。

用法：python tools/make_fig5_gamma_symbolic.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "figures", "fig5_gamma_symbolic.png")

SS = 2                      # 超采样倍数
FW, FH = 1900, 1050         # 最终尺寸
W, H = FW * SS, FH * SS

FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size * SS)


# 配色（与 fig1-4 一致的浅色系）
C_BG = "#ffffff"
C_LEFT = ("#e8f1fb", "#4a7fb5")     # LLM 事件流：蓝
C_HT = ("#fbe9e7", "#c05b4d")       # 5-HT：粉
C_OA = ("#fdf3e0", "#c8862a")       # OA：橙
C_DA = ("#e9f5e9", "#4d8a4d")       # DA：绿
C_SNN = ("#f0eaf9", "#7a5fa8")      # SNN：紫
C_TXT = "#222222"
C_SUB = "#555555"
C_BAND = ("#f2f4f7", "#8a94a6")


def box(dr, xy, fill, outline, radius=18 * SS, width=3):
    dr.rounded_rectangle(xy, radius=radius, fill=fill,
                         outline=outline, width=width * SS)


def text_block(dr, cx, y, lines, center=True):
    """lines: [(text, font, color, dy)]，返回结束 y。"""
    for t, ft, color, dy in lines:
        bb = dr.textbbox((0, 0), t, font=ft)
        tw = bb[2] - bb[0]
        dr.text((cx - tw // 2, y), t, font=ft, fill=color)
        y += dy * SS
    return y


def arrow(dr, p0, p1, color, width=3, head=16):
    dr.line([p0, p1], fill=color, width=width * SS)
    import math
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    h = head * SS
    a1 = (p1[0] - h * math.cos(ang - 0.42), p1[1] - h * math.sin(ang - 0.42))
    a2 = (p1[0] - h * math.cos(ang + 0.42), p1[1] - h * math.sin(ang + 0.42))
    dr.polygon([p1, a1, a2], fill=color)


def main():
    img = Image.new("RGB", (W, H), C_BG)
    dr = ImageDraw.Draw(img)

    # 标题
    t = "Neuromodulation Vector gamma = [DA, OA, 5-HT] in the LeLamp Control Loop"
    bb = dr.textbbox((0, 0), t, font=f(30, True))
    dr.text(((W - (bb[2] - bb[0])) // 2, 30 * SS), t, font=f(30, True), fill=C_TXT)
    t2 = "Symbolic form -- coefficients theta are the Phase-2 (Level A) evolution targets"
    bb = dr.textbbox((0, 0), t2, font=f(19))
    dr.text(((W - (bb[2] - bb[0])) // 2, 78 * SS), t2, font=f(19), fill=C_SUB)

    # 左侧：LLM 事件流
    lx0, ly0, lx1, ly1 = 60, 340, 420, 700
    box(dr, (lx0 * SS, ly0 * SS, lx1 * SS, ly1 * SS), *C_LEFT)
    text_block(dr, (lx0 + lx1) // 2 * SS, (ly0 + 55) * SS, [
        ("LLM event stream", f(22, True), C_TXT, 46),
        ("each event carries", f(17), C_SUB, 34),
        ("gamma = [DA, OA, 5-HT]", f(17), C_SUB, 34),
        ("per emotional state,", f(17), C_SUB, 34),
        ("switched online", f(17), C_SUB, 34),
        ("(engineering analogy)", f(14), C_SUB, 30),
    ])

    # 中间三条通路
    mx0, mx1 = 620, 1230
    paths = [
        ("5-HT  ->  leak time constant",
         "leak = clip( l0 + l1 * (5HT - 0.5) )",
         "sluggish  <->  agile dynamics", C_HT, 130, 330),
        ("OA  ->  sensory injection gain",
         "gain = g0 + g1 * OA",
         "arousal: amplifies error signals", C_OA, 400, 600),
        ("DA  ->  three-factor learning gate",
         "gate = d0 + d1 * DA",
         "W <- W + eta * gate * delta * E", C_DA, 670, 900),
    ]
    for title, formula, sub, col, py0, py1 in paths:
        box(dr, (mx0 * SS, py0 * SS, mx1 * SS, py1 * SS), *col)
        text_block(dr, (mx0 + mx1) // 2 * SS, (py0 + 28) * SS, [
            (title, f(20, True), C_TXT, 44),
            (formula, f(19), C_TXT, 40),
            (sub, f(15), C_SUB, 32),
        ])

    # 右侧：SNN + DN 读出
    rx0, ry0, rx1, ry1 = 1420, 340, 1840, 700
    box(dr, (rx0 * SS, ry0 * SS, rx1 * SS, ry1 * SS), *C_SNN)
    text_block(dr, (rx0 + rx1) // 2 * SS, (ry0 + 45) * SS, [
        ("FlyBrain SNN + DN readout", f(21, True), C_TXT, 46),
        ("h <- tanh(leak*h + msg", f(17), C_TXT, 34),
        ("        + gain*x + b)", f(17), C_TXT, 40),
        ("500 nodes, edges frozen", f(16), C_SUB, 34),
        ("dq = W * s_DN", f(17, True), C_TXT, 36),
        ("-> safety layer -> servos", f(16), C_SUB, 34),
    ])

    # 箭头：左 -> 三通路 -> 右
    for _, _, _, col, py0, py1 in paths:
        cy = (py0 + py1) // 2 * SS
        arrow(dr, (lx1 * SS, 520 * SS), (mx0 * SS, cy), col[1])
        arrow(dr, (mx1 * SS, cy), (rx0 * SS, 520 * SS), col[1])

    # 底部信息带
    by0, by1 = 950, 1030
    box(dr, (60 * SS, by0 * SS, 1840 * SS, by1 * SS), *C_BAND, radius=12 * SS, width=2)
    text_block(dr, W // 2, (by0 + 10) * SS, [
        ("theta = (l0, l1, g0, g1, d0, d1)   |   Phase 2 (Level A): outer-loop CMA-ES evolves theta; "
         "inner three-factor learning unchanged", f(15, True), C_TXT, 30),
        ("gamma changes HOW the same fixed topology runs, never its wiring.  "
         "Engineering analogy, not a biological equivalence claim.", f(14), C_SUB, 26),
    ])

    img = img.resize((FW, FH), Image.LANCZOS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()

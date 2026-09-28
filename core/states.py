# -*- coding: utf-8 -*-
"""六种果蝇内部状态定义（继承 gemini 设计，作为 T1 验证真值 / 行为先验）。

姿态基元 = 相对 SIM_ZERO 的关节偏移 + 可选振荡波形。
所有偏移均已校验不越过关节限位（SafetyLayer 仍会二次钳制）。
"""
import numpy as np

from configs.default_config import SIM_ZERO

STATES = {
    "Emotional_Comfort": {
        "gamma": [0.3, 0.1, 0.9],
        "pose_offset": [0.0, 0.55, -0.45, 0.0, 0.45],     # 前倾下倾
        "osc": {"joints": [1, 2], "amp": 0.05, "freq": 0.3},   # 0.3Hz 呼吸律动
        "led_rgb": [1.0, 0.6, 0.8], "led_mode": "slow_breathing",
        "text": "I am here for you.",
    },
    "Fear_Arousal": {
        "gamma": [0.8, 0.9, 0.2],
        "pose_offset": [0.0, -0.45, -0.55, 0.0, -0.25],   # 后撤高抬
        "osc": {"joints": [0, 1, 2, 3, 4], "amp": 0.03, "freq": 8.0},  # 微幅震颤
        "led_rgb": [1.0, 0.1, 0.1], "led_mode": "fast_strobe",
        "text": "Watch out! Obstacle ahead.",
    },
    "Approach_Reward": {
        "gamma": [0.8, 0.2, 0.5],
        "pose_offset": [0.0, 0.50, -0.30, 0.0, 0.30],
        "osc": {"joints": [0], "amp": 0.20, "freq": 0.8},  # 欢快摆头
        "led_rgb": [1.0, 0.85, 0.3], "led_mode": "steady",
        "text": "Glad to be with you.",
    },
    "Vigilance_Defense": {
        "gamma": [0.2, 0.5, 0.9],
        "pose_offset": [0.0, 0.10, -0.10, 0.0, 0.10],     # 立直
        "osc": {"joints": [0], "amp": 0.80, "freq": 0.15}, # 缓慢环视扫视
        "led_rgb": [0.4, 0.7, 1.0], "led_mode": "steady",
        "text": "Scanning surroundings...",
    },
    "Exploration_Seeking": {
        "gamma": [0.5, 0.5, 0.4],
        "pose_offset": [0.0, 0.40, -0.60, 0.0, 0.55],     # 低头巡航
        "osc": {"joints": [0], "amp": 0.45, "freq": 0.4},  # S 型弧线（偏航扫描近似）
        "led_rgb": [0.3, 1.0, 0.4], "led_mode": "alt_breathing",
        "text": "Exploring your workspace.",
    },
    "Grooming_Rest": {
        "gamma": [0.1, 0.1, 0.8],
        "pose_offset": [0.0, 0.25, -0.95, 0.0, 0.85],     # 收拢垂首休眠
        "osc": {"joints": [], "amp": 0.0, "freq": 0.0},
        "led_rgb": [0.1, 0.1, 0.4], "led_mode": "dim",
        "text": "",
    },
}

STATE_NAMES = list(STATES.keys())


def state_onehot(state_name, n_sensory):
    """状态 -> 感觉节点驱动向量（one-hot 重复铺满感觉节点）。"""
    idx = STATE_NAMES.index(state_name)
    x = np.zeros(n_sensory)
    x[idx % n_sensory] = 1.0
    x[(idx + len(STATE_NAMES)) % n_sensory] = 0.5
    return x


def build_sensory_input(state_name, q, q_target, n_sensory, err_gain=3.0):
    """完整感觉输入（ON/OFF 整流通道设计，仿果蝇视觉 ON/OFF 通路）：
    - 通道 0-4  ：ON 误差通道  err_gain * max(+err, 0)（pre-DN 单跳直达）
    - 通道 5-9  ：OFF 误差通道 err_gain * max(-err, 0)（pre-DN 单跳直达）
    - 通道 12+  ：状态 one-hot（多跳扩散，提供情境门控）

    relu 放电率 s>=0 无法携带符号，正负误差必须分通道编码，
    否则 LMS/三因子学到单侧 runaway（2026-09-28 实证）。
    """
    x = np.zeros(n_sensory)
    idx = STATE_NAMES.index(state_name)
    x[12 + idx % max(1, n_sensory - 12)] = 1.0
    err = np.clip((q_target - q) / 1.0, -1.0, 1.0)
    for j in range(5):
        x[j] = err_gain * max(err[j], 0.0)
        x[5 + j] = err_gain * max(-err[j], 0.0)
    return x


def pose_base(state_name):
    """静态姿态基元（不含振荡）：T1 训练目标。振荡属演示层调制，不作为学习真值。"""
    return SIM_ZERO + np.asarray(STATES[state_name]["pose_offset"], dtype=float)


def pose_primitive(state_name, t):
    """状态姿态基元：q_target(t) = SIM_ZERO + offset + osc(t)。"""
    st = STATES[state_name]
    q = SIM_ZERO + np.asarray(st["pose_offset"], dtype=float)
    osc = st["osc"]
    if osc["amp"] > 0 and osc["joints"]:
        wave = osc["amp"] * np.sin(2.0 * np.pi * osc["freq"] * t)
        for j in osc["joints"]:
            q[j] += wave
    return q


def led_rgba(state_name, t):
    """LED 光效：按模式生成 RGBA。"""
    st = STATES[state_name]
    rgb = st["led_rgb"]
    mode = st["led_mode"]
    if mode == "slow_breathing":
        a = 0.55 + 0.40 * np.sin(2.0 * np.pi * (1.0 / 3.0) * t)   # 3s 周期
    elif mode == "fast_strobe":
        a = 1.0 if np.sin(2.0 * np.pi * 10.0 * t) > 0 else 0.15   # 10Hz 爆闪
    elif mode == "alt_breathing":
        a = 0.55 + 0.40 * np.sin(2.0 * np.pi * 0.5 * t)
    elif mode == "dim":
        a = 0.25
    else:  # steady
        a = 1.0
    return [rgb[0], rgb[1], rgb[2], float(np.clip(a, 0.05, 1.0))]

# -*- coding: utf-8 -*-
"""FlyBrain SNN 引擎：率模型循环网络 + 神经调控 + 稳态。

控制器方程（与 FlyDoom 一致）：
    h_next = activation(leak * h + sparse_message_passing(h, E, w) + x + b)

神经调控向量 gamma = [DA, OA, 5-HT] 的工程作用通道（类比，非生物声明）：
- DA   : 三因子学习第三因子增益（在 dn_readout 中生效）
- OA   : 感觉注入增益（唤醒水平）
- 5-HT : leak 时间常数调制（静-动谱系）
稳态：homeostatic bias 使平均活动保持在目标附近，保证有界。
"""
import numpy as np

from configs.default_config import LEAK_BASE


class FlyBrain:
    def __init__(self, graph, seed=42, target_rate=0.30, homeo_lr=1e-3, mod=None):
        self.graph = graph
        self.rng = np.random.default_rng(seed + 1)
        self.h = np.zeros(graph.n)
        self.b0 = self.rng.normal(0.0, 0.05, size=graph.n)  # 初始偏置（reset 基准）
        self.b = self.b0.copy()                              # 稳态偏置
        self.target_rate = target_rate
        self.homeo_lr = homeo_lr
        # γ 映射系数 θ（Phase-2 Level A 进化目标；默认值 = Phase-1 手工先验，
        # 不传 mod 时行为与阶段一完全一致）
        self.mod = {"leak_base": LEAK_BASE, "leak_amp": 0.10,
                    "gain_bias": 0.5, "gain_slope": 1.5}
        if mod:
            self.mod.update(mod)

    def reset(self):
        """episode 边界完整重置循环状态（FlyDoom 惯例）：
        h 与 homeostatic bias 都复位——b 是循环状态的一部分，
        否则训练在收敛后的 b 上学 W、评估用初始 b，二者符号结构失配
        （2026-09-28 实证：训练收敛、确定性评估撞墙发散的根因）。
        """
        self.h[:] = 0.0
        self.b[:] = self.b0

    def step(self, x_sensory, gamma):
        """推进一步。

        x_sensory: shape (n_sensory,)，只注入被指定的感觉节点（接口纪律）。
        gamma: [DA, OA, 5-HT] in [0,1]^3
        返回 s: shape (n,)，整网放电率（relu(h)）。
        """
        da, oa, ht = [float(np.clip(g, 0.0, 1.0)) for g in gamma]

        m = self.mod
        # 5-HT 调制时间常数 / OA 调制唤醒增益（系数为 Level A 进化目标；
        # leak 硬边界放宽至 [0.40, 0.98] 以允许外环探索，默认 θ 输出范围不变）
        leak = np.clip(m["leak_base"] + m["leak_amp"] * (ht - 0.5), 0.40, 0.98)
        gain = m["gain_bias"] + m["gain_slope"] * oa

        x = np.zeros(self.graph.n)
        x[self.graph.sensory_idx] = gain * np.asarray(x_sensory)

        msg = self.graph.message_passing(np.maximum(self.h, 0.0))
        self.h = np.tanh(leak * self.h + msg + x + self.b)

        s = np.maximum(self.h, 0.0)

        # 稳态偏置：保持平均活动有界
        self.b += self.homeo_lr * (self.target_rate - s.mean())

        return s

    def dn_rates(self, s):
        """支柱二接口：只有 DN 节点可被读出。"""
        return s[self.graph.dn_idx]

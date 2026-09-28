# -*- coding: utf-8 -*-
"""支柱二 + 支柱三：下降神经元固定读出 与 资格迹 x 多巴胺三因子学习。

读出（支柱二）：
    dq = W_DN @ s_DN        W_DN in R^{5 x N_DN}，只有 DN 节点接入读出（纪律）。

三因子学习（支柱三，无反向传播、无优化器）：
    E = gamma_trace * E + outer(dq_norm, s_DN)     # 逐突触资格迹（局部前后活动）
    W += eta * (DA * delta) * E                     # 第三因子 = DA 调制的 TD 误差
价值读出（局部 delta 规则）：
    V = w_v @ s_DN ;  delta = r + disc * V' - V ;  w_v += v_lr * delta * s_DN
"""
import numpy as np

from configs.default_config import (
    GAMMA_TRACE, ETA_W, W_MAX, W_DECAY, V_LR, DISCOUNT, MAX_DQ_PER_STEP,
)


class DNReadout:
    def __init__(self, n_dn=10, n_joints=5, seed=42, trainable=True):
        rng = np.random.default_rng(seed + 2)
        self.n_dn = n_dn
        self.n_joints = n_joints
        self.W = rng.normal(0.0, 0.05, size=(n_joints, n_dn))
        self.E = np.zeros_like(self.W)          # 资格迹
        self.w_v = np.zeros(n_dn)               # 价值读出
        self.trainable = trainable
        self.last_delta = 0.0
        self.s_ema = None                       # DN 活动慢速滑动均值（共模抑制）
        self.ema_alpha = 0.99
        self.use_center = False                 # 2026-09-28：误差通道改 pre-DN 单跳后，
                                                # 共模抑制会滤除伺服所需的低频误差分量，停用
        self.d_var = 1.0                        # TD 误差滑动方差（δ 归一化）
        self._last_centered = np.zeros(n_dn)
        self.da_floor = 0.2                     # DA 门控下限（Phase-2 Level A 进化目标 θ）
        self.da_slope = 0.8                     # DA 门控斜率（默认 = Phase-1 手工先验）

    def _center(self, s_dn):
        """共模抑制（可选，use_center=True 时启用）：减去慢速 EMA。"""
        s_dn = np.asarray(s_dn, dtype=float)
        if not self.use_center:
            self._last_centered = s_dn
            return s_dn
        if self.s_ema is None:
            self.s_ema = s_dn.copy()
        centered = s_dn - self.s_ema
        self.s_ema = self.ema_alpha * self.s_ema + (1 - self.ema_alpha) * s_dn
        self._last_centered = centered
        return centered

    def act(self, s_dn):
        """DN 放电率（共模抑制后）-> 5 维关节增量（rad），已含单步限幅尺度。"""
        dq = self.W @ self._center(s_dn)
        return np.clip(dq, -MAX_DQ_PER_STEP, MAX_DQ_PER_STEP)

    def value(self, s_dn=None):
        """价值读出：使用最近一次 act 计算的中心化活动（避免重复推进 EMA）。"""
        s = self._last_centered if s_dn is None else self._center(s_dn)
        return float(np.clip(self.w_v @ s, -10.0, 10.0))

    def update(self, s_dn, dq, reward, v_next, da, err_vec=None, mode="hybrid"):
        """学习更新（无反向传播）。mode: hybrid | lms | three_factor

        通道 1 —— 三因子（three_factor/hybrid）：eta * (DA * delta) * E。
        通道 2 —— 局部 delta 规则 LMS（lms/hybrid）：eta_d * outer(err, s_DN)，DA 门控。
        实证记录（2026-09-28）：hybrid 下两通道偶发互搏（W 在伺服正确/反向间摆动）；
        纯 three_factor 在 T1 上 300 episodes 无可测学习（负结果）；
        纯 lms 为当前 T1 默认收敛路径（见 train_t1 --learn）。
        """
        s_dn = self._last_centered
        v = self.value()
        delta = reward + DISCOUNT * v_next - v
        delta = float(np.clip(delta, -10.0, 10.0))   # TD 误差硬限幅（防价值发散）
        self.last_delta = delta

        # 价值读出：局部 delta 规则 + 权重硬边界
        self.w_v += V_LR * delta * s_dn
        np.clip(self.w_v, -50.0, 50.0, out=self.w_v)

        if self.trainable:
            da_gate = self.da_floor + self.da_slope * float(np.clip(da, 0, 1))

            # δ 归一化（滑动方差）：抑制奖励尺度漂移引发的发散
            self.d_var = 0.99 * self.d_var + 0.01 * delta ** 2
            delta_n = delta / (np.sqrt(self.d_var) + 1e-3)
            delta_n = float(np.clip(delta_n, -3.0, 3.0))

            # 资格迹：outer(动作分量, 突触前活动)，限幅防单步爆冲
            dq_norm = dq / MAX_DQ_PER_STEP
            self.E = GAMMA_TRACE * self.E + np.outer(dq_norm, s_dn)
            np.clip(self.E, -1.0, 1.0, out=self.E)

            # 通道 1：三因子
            if mode in ("hybrid", "three_factor"):
                self.W += ETA_W * da_gate * delta_n * self.E

            # 通道 2：误差引导局部 delta 规则（DA 门控），系数 0.5 抑制回合内摆幅
            if mode in ("hybrid", "lms") and err_vec is not None:
                e = np.clip(np.asarray(err_vec, dtype=float), -1.0, 1.0)
                self.W += ETA_W * 0.5 * da_gate * np.outer(e, s_dn)

            # 突触稳态：向 0 收缩 + 硬边界（FlyDoom homeostatic bias 与权重边界）
            self.W *= (1.0 - W_DECAY)
            np.clip(self.W, -W_MAX, W_MAX, out=self.W)

    def save(self, path):
        np.savez(path, W=self.W, w_v=self.w_v,
                 s_ema=self.s_ema if self.s_ema is not None else np.zeros(self.n_dn))

    def load(self, path):
        dat = np.load(path)
        self.W = dat["W"]
        self.w_v = dat["w_v"]
        if "s_ema" in dat:
            self.s_ema = dat["s_ema"]

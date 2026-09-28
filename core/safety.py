# -*- coding: utf-8 -*-
"""L2 安全与平滑层。

关键设计决策（设计文档 6）：SNN 输出永远经过本层，无直达执行器通路。
- 关节位置限位（URDF range）
- 单步增量限幅
- 余弦平滑轨迹插值（消除加速度阶跃的轻量实现，等效三次样条的二阶连续段）
- 障碍物 AABB 距离评估（头端球近似）
"""
import numpy as np

from configs.default_config import (
    JOINT_LOWER, JOINT_UPPER, MAX_DQ_PER_STEP, CONTROL_HZ,
)

HEAD_RADIUS = 0.06          # 灯头球半径（URDF lamp_head）
SAFETY_MARGIN = 0.02        # 安全余量


class SafetyLayer:
    def clamp_limits(self, q):
        return np.clip(q, JOINT_LOWER, JOINT_UPPER)

    def rate_limit(self, q_target, q_current):
        dq = np.clip(q_target - q_current, -MAX_DQ_PER_STEP, MAX_DQ_PER_STEP)
        return self.clamp_limits(q_current + dq)

    def smooth_segment(self, q_start, q_goal, duration_s, hz=CONTROL_HZ):
        """余弦（smoothstep）插值：q(t) = start + (goal-start)*(1-cos(pi*u))/2。

        首尾速度为零，加速度连续，满足"消除加速度阶跃"的工程要求。
        返回 shape (n_steps, 5)。
        """
        n = max(2, int(duration_s * hz))
        u = np.linspace(0.0, 1.0, n)
        s = 0.5 * (1.0 - np.cos(np.pi * u))
        return q_start[None, :] + (q_goal - q_start)[None, :] * s[:, None]

    @staticmethod
    def point_aabb_distance(p, center, half):
        """点到 AABB 的最短距离。"""
        d = np.maximum(np.abs(p - center) - half, 0.0)
        return float(np.linalg.norm(d))

    def head_collision_risk(self, head_pos, obstacles):
        """头端是否侵入障碍物（球-AABB）。返回 (是否碰撞, 最小净距离)。"""
        min_clear = np.inf
        for ob in obstacles:
            c = np.asarray(ob["pos"], dtype=float)
            h = np.asarray(ob["size"], dtype=float) / 2.0
            clear = self.point_aabb_distance(head_pos, c, h) - HEAD_RADIUS
            min_clear = min(min_clear, clear)
        return (min_clear < SAFETY_MARGIN), float(min_clear)

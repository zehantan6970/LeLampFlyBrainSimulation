# -*- coding: utf-8 -*-
"""T1 状态表达任务：三因子学习训练 DN 读出（无反向传播、无优化器）。

每个 episode 随机抽一种内部状态，注入对应感觉编码与 gamma，
奖励 = -R_POSE_SCALE * 姿态误差 + 进入邻域奖励 + 存活成本，
三因子规则更新 W_DN；图边集永不变动。

用法：
    python experiments/train_t1.py --episodes 200 --seed 42 --graph mock
    python experiments/train_t1.py --graph er --seed 1     # 消融臂
输出：
    data/t1_train_{graph}_seed{seed}.csv   每 episode 奖励曲线
    data/w_dn_t1.npz                        检查点（供 main.py --controller brain）
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import (  # noqa: E402
    CONTROL_HZ, SNN_STEPS_PER_CONTROL, SIM_ZERO, N_DN, N_SENSORY,
    R_ALIVE, R_POSE_SCALE, R_NEAR_BONUS, NEAR_THRESHOLD,
    CHECKPOINT_PATH, MAX_DQ_PER_STEP,
)

MAX_DQ = MAX_DQ_PER_STEP
from core.connectome import ConnectomeGraph  # noqa: E402
from core.fly_brain import FlyBrain  # noqa: E402
from core.dn_readout import DNReadout  # noqa: E402
from core.states import (  # noqa: E402
    STATES, STATE_NAMES, state_onehot, pose_base, build_sensory_input,
)
from core.safety import SafetyLayer  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_episode(brain, readout, safety, state_name, rng, noise_std, mode, seconds=4.0):
    """单 episode：给定状态，学习让 DN 读出复现其静态姿态基元。

    探索机制（三因子框架的必要成分）：在 DN 读出动作上叠加高斯噪声，
    噪声进入实际执行的 dq，因此被资格迹正确归因——奖励改善的方向被强化。
    """
    brain.reset()
    q = SIM_ZERO.copy()
    gamma = STATES[state_name]["gamma"]
    da = gamma[0]
    q_target = pose_base(state_name)

    err_prev = float(np.max(np.abs(q - q_target)))
    ou = np.zeros(5)                       # OU 探索噪声状态
    total_r = 0.0
    steps = int(seconds * CONTROL_HZ)
    for k in range(steps):
        x = build_sensory_input(state_name, q, q_target, N_SENSORY)
        s = None
        for _ in range(SNN_STEPS_PER_CONTROL):
            s = brain.step(x, gamma)
        s_dn = brain.dn_rates(s)

        # OU 时序相关探索噪声：白噪声扩散无法形成持续位移，相关噪声才能"走远"
        ou = 0.92 * ou + 0.25 * rng.normal(0.0, noise_std, size=5)
        dq = readout.act(s_dn) + ou * MAX_DQ
        dq = np.clip(dq, -MAX_DQ, MAX_DQ)
        q = safety.rate_limit(q + dq, q)

        err = float(np.max(np.abs(q - q_target)))
        # 进展奖励：误差下降即有正奖励，梯度方向明确（密集塑形）
        r = R_ALIVE + 8.0 * (err_prev - err) \
            + (R_NEAR_BONUS if err < NEAR_THRESHOLD else 0.0)
        err_prev = err

        v_next = readout.value()
        readout.update(s_dn, dq, r, v_next, da, err_vec=(q_target - q), mode=mode)
        total_r += r
    # 终端误差惩罚：防止"无进展但不亏"的停滞解
    total_r += -R_POSE_SCALE * err_prev
    return total_r / steps


def evaluate_deterministic(graph, readout, safety):
    """确定性评估（无噪声、W 冻结）：六状态末端最大关节误差。用于检查点选择。

    工程必要性（2026-09-28 实证）：三因子/LMS 更新在回合末段可能失稳，
    训练滚动奖励与最终权重表现可严重背离；必须以确定性评估选最优检查点。
    """
    errs = []
    for st in STATE_NAMES:
        brain = FlyBrain(graph, seed=42)
        brain.reset()
        q = SIM_ZERO.copy()
        qt = pose_base(st)
        gamma = STATES[st]["gamma"]
        for k in range(int(4.0 * CONTROL_HZ)):
            x = build_sensory_input(st, q, qt, N_SENSORY)
            s = None
            for _ in range(SNN_STEPS_PER_CONTROL):
                s = brain.step(x, gamma)
            dq = readout.act(brain.dn_rates(s))
            q = safety.rate_limit(q + dq, q)
        errs.append(float(np.max(np.abs(q - qt))))
    return float(np.mean(errs)), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--graph", choices=["mock", "er", "rewired"], default="mock")
    ap.add_argument("--learn", choices=["hybrid", "lms", "three_factor"], default="lms",
                    help="lms=局部delta规则(T1默认收敛路径); three_factor=纯三因子(科学对照); hybrid=双通道")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed + 100)
    graph = ConnectomeGraph.build(seed=args.seed, variant=args.graph)
    brain = FlyBrain(graph, seed=args.seed)
    readout = DNReadout(n_dn=N_DN, n_joints=5, seed=args.seed, trainable=True)
    safety = SafetyLayer()

    print(f"[t1] graph={args.graph} edges={graph.meta['n_edges']} seed={args.seed} "
          f"episodes={args.episodes}")

    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    csv_path = os.path.join(PROJECT_ROOT, "data",
                            f"t1_train_{args.graph}_seed{args.seed}.csv")
    t0 = time.time()
    rewards = []
    ckpt = os.path.join(PROJECT_ROOT, CHECKPOINT_PATH)
    best = {"err": np.inf, "ep": -1}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "state", "mean_reward", "td_delta", "elapsed_s"])
        for ep in range(args.episodes):
            state_name = STATE_NAMES[rng.integers(0, len(STATE_NAMES))]
            noise_std = max(0.1, 0.6 * (0.995 ** ep))   # 探索噪声退火
            mr = run_episode(brain, readout, safety, state_name, rng, noise_std,
                             args.learn)
            rewards.append(mr)
            w.writerow([ep, state_name, f"{mr:.4f}",
                        f"{readout.last_delta:.4f}", f"{time.time()-t0:.1f}"])
            if (ep + 1) % 20 == 0:
                recent = np.mean(rewards[-20:])
                eval_err, _ = evaluate_deterministic(graph, readout, safety)
                marker = ""
                if eval_err < best["err"]:
                    best = {"err": eval_err, "ep": ep + 1}
                    readout.save(ckpt)
                    marker = "  <- 新最优，已保存检查点"
                print(f"[t1] ep {ep+1:>4d}/{args.episodes}  近20轮奖励 = {recent:+.4f}"
                      f"  确定性评估误差 = {eval_err:.3f}{marker}")

    first, last = np.mean(rewards[:20]), np.mean(rewards[-20:])
    print(f"[t1] 完成。前20轮均值 {first:+.4f} -> 末20轮均值 {last:+.4f} "
          f"（提升 {'是' if last > first else '否'}）")
    print(f"[t1] 最优检查点：ep {best['ep']}，确定性评估误差 {best['err']:.3f}")
    print(f"[t1] 奖励曲线：{csv_path}")
    print(f"[t1] 检查点：{ckpt}")


if __name__ == "__main__":
    main()

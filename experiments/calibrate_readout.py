# -*- coding: utf-8 -*-
"""DN 读出上界基线：批量岭回归标定（非生物学习规则，作 upper-bound 对照）。

设计文档 4.5 定位：手工/工程标定的 W_DN 是「学习结果应逼近的上界」。
本脚本采集 (s_DN, dq_desired) 样本对，求解 W = Dq S^T (S S^T + λI)^{-1}。
不经过任何网络反向传播；连接组边集与内部权重不受影响。

用法：
    python experiments/calibrate_readout.py --samples 4000 --seed 42
输出：
    data/w_dn_calibrated.npz   标定读出（main.py --controller brain --ckpt 指定）
    并打印六状态确定性评估误差（应与 zero-W 基线相比显著下降）
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import (  # noqa: E402
    CONTROL_HZ, SNN_STEPS_PER_CONTROL, SIM_ZERO, N_DN, N_SENSORY,
    MAX_DQ_PER_STEP, JOINT_LOWER, JOINT_UPPER,
)
from core.connectome import ConnectomeGraph  # noqa: E402
from core.fly_brain import FlyBrain  # noqa: E402
from core.states import STATES, STATE_NAMES, pose_base, build_sensory_input  # noqa: E402
from core.dn_readout import DNReadout  # noqa: E402
from core.safety import SafetyLayer  # noqa: E402
from experiments.train_t1 import evaluate_deterministic  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect(graph, n_samples, seed):
    """随机扰动 q，采集 DN 活动与期望修正步（比例伺服律 dq=clip(0.25*err)）。"""
    rng = np.random.default_rng(seed)
    safety = SafetyLayer()
    S, D = [], []
    for st in STATE_NAMES:
        qt = pose_base(st)
        gamma = STATES[st]["gamma"]
        per = n_samples // len(STATE_NAMES)
        brain = FlyBrain(graph, seed=42)
        brain.reset()
        q = SIM_ZERO.copy()
        for i in range(per):
            # 周期性随机扰动 q，覆盖误差空间
            if i % 20 == 0:
                span = (JOINT_UPPER - JOINT_LOWER)
                q = np.clip(SIM_ZERO + rng.uniform(-0.3, 0.3, 5) * span,
                            JOINT_LOWER, JOINT_UPPER)
            x = build_sensory_input(st, q, qt, N_SENSORY)
            s = None
            for _ in range(SNN_STEPS_PER_CONTROL):
                s = brain.step(x, gamma)
            s_dn = brain.dn_rates(s)
            dq_des = np.clip(0.25 * (qt - q), -MAX_DQ_PER_STEP, MAX_DQ_PER_STEP)
            S.append(s_dn)
            D.append(dq_des)
            q = safety.rate_limit(q + dq_des, q)
    return np.asarray(S), np.asarray(D)


def solve_small(A, B):
    """纯 numpy 高斯消元解 AX=B（A n*n 小矩阵）。

    本机 numpy linalg.inv/solve 调用 MKL LAPACK 时被应用控制策略静默拦截
    （2026-09-28 实测 exit 127 无回溯），10 阶系统用手写消元规避。
    """
    n = A.shape[0]
    M = np.concatenate([A.astype(float).copy(), B.astype(float).copy()], axis=1)
    for col in range(n):
        piv = col + int(np.argmax(np.abs(M[col:, col])))
        if abs(M[piv, col]) < 1e-12:
            raise np.linalg.LinAlgError("singular")
        M[[col, piv]] = M[[piv, col]]
        M[col] /= M[col, col]
        for r in range(n):
            if r != col:
                M[r] -= M[r, col] * M[col]
    return M[:, n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam", type=float, default=1e-3)
    args = ap.parse_args()

    graph = ConnectomeGraph.build(seed=args.seed, variant="mock")
    S, D = collect(graph, args.samples, args.seed)
    print(f"[calib] 样本 {S.shape[0]} 组，s_DN 均值 {S.mean():.3f}")

    # 岭回归：W = D S^T (S S^T + λI)^{-1}
    # 注意：本机 MKL GEMM 被应用控制策略静默拦截（2026-09-28 实测），
    # 故用广播+sum 替代大矩阵乘；10 维系统开销可忽略。
    StS = (S[:, :, None] * S[:, None, :]).sum(axis=0)
    print("[calib] StS done", flush=True)
    DtS = (D[:, :, None] * S[:, None, :]).sum(axis=0)   # (5, 10)
    print("[calib] DtS done", flush=True)
    A = StS + args.lam * np.eye(S.shape[1])
    Ainv = solve_small(A, np.eye(S.shape[1]))
    print("[calib] solve done", flush=True)
    W = np.einsum("ij,jk->ik", DtS, Ainv)   # 规避 GEMM 路径
    print("[calib] W done", flush=True)

    readout = DNReadout(n_dn=N_DN, n_joints=5, seed=args.seed, trainable=False)
    readout.W = W
    safety = SafetyLayer()
    mean_err, errs = evaluate_deterministic(graph, readout, safety)
    print(f"[calib] 确定性评估 per-state 误差: "
          + " ".join(f"{e:.3f}" for e in errs)
          + f" | mean={mean_err:.3f}")
    print("[calib] 参考：zero-W（静止）基线 mean=0.542；越低越好")

    out = os.path.join(PROJECT_ROOT, "data", "w_dn_calibrated.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    readout.save(out)
    print(f"[calib] 已保存：{out}")
    print("[calib] 用以下命令查看脑控制器演示：")
    print("  python main.py --mode demo --controller brain --ckpt data/w_dn_calibrated.npz")


if __name__ == "__main__":
    main()

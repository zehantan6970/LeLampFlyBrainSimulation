# -*- coding: utf-8 -*-
"""Phase-2 Level A：外环 CMA-ES 进化 γ 映射系数 θ，内环三因子学习不变（双层优化）。

θ = (leak_base, leak_amp, gain_bias, gain_slope, da_floor, da_slope)
    分别对应 fly_brain.step 的 leak = clip(l0 + l1·(5HT−0.5))、gain = g0 + g1·OA
    与 dn_readout.update 的 da_gate = d0 + d1·DA。

双层结构（生物学同构：演化调调制、发育调突触）：
  外环 —— CMA-ES 搜索 θ（在归一化 [0,1]^6 空间优化，再映射到物理边界）；
  内环 —— FlyDoom 三因子学习（资格迹 × DA 门控 TD 误差），规则本身不变。

适应度（最小化，全部分量落盘供事后重加权分析）：
  fitness = 1.0·终态确定性误差均值 + 0.5·误差跨种子标准差
          − 1.0·学习曲线增益（末25%奖励 − 初25%奖励）
          + 2.0·平滑度惩罚（确定性评估的平均 |Δdq|）
          + 约束惩罚
防作弊约束（2026-09-28 设计文档 P2-4 预案）：
  da_floor + da_slope < 0.35  → +1.0（学习门控退化："不学"套利）
  gain_slope < 0.05           → +0.5（感觉通路退化："闭眼"套利）
  适应度必须含学习增益项，否则外环可用冻结学习换取终态误差方差。

工程约束（本机 2026-09-28）：np.linalg.*(LAPACK) 与大矩阵 GEMM 被策略静默拦截
（exit 127 无回溯）。CMA-ES 协方差特征分解改用手写 Jacobi 旋转（6×6 对称矩阵，
纯 elementwise 运算）；全部矩阵运算 ≤6 维，不触达崩溃路径。

数据落盘（两级，格式见设计文档）：
  level 1  data/evolve_modulation.jsonl         每次 θ 评估一行（适应度全部分量）
  level 2  data/evolve_curves_g{gen}_i{idx}.csv 该 θ 内环训练逐 episode 奖励曲线
  最优 θ   data/modulation_theta_best.json      每代刷新（真实物理单位）

用法：
  # 冒烟验证（约 3–6 分钟）
  python experiments/evolve_modulation.py --popsize 4 --generations 2 --episodes 6 --seeds 1
  # 正式运行（默认预算约数小时，建议过夜）
  python experiments/evolve_modulation.py --popsize 6 --generations 12 --episodes 40 --seeds 2
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.default_config import (  # noqa: E402
    CONTROL_HZ, SNN_STEPS_PER_CONTROL, SIM_ZERO, N_DN, N_SENSORY,
)
from core.connectome import ConnectomeGraph  # noqa: E402
from core.fly_brain import FlyBrain  # noqa: E402
from core.dn_readout import DNReadout  # noqa: E402
from core.states import STATES, STATE_NAMES, pose_base, build_sensory_input  # noqa: E402
from core.safety import SafetyLayer  # noqa: E402
from experiments.train_t1 import run_episode  # noqa: E402

# ---------------- θ 定义：归一化空间 [0,1]^6 <-> 物理边界 ----------------
THETA_KEYS = ["leak_base", "leak_amp", "gain_bias", "gain_slope", "da_floor", "da_slope"]
BOUNDS = [                               # (lo, hi)，默认值为阶段一手工先验
    (0.55, 0.90),                        # leak_base  （先验 0.70）
    (0.00, 0.60),                        # leak_amp   （先验 0.10）
    (0.00, 1.50),                        # gain_bias  （先验 0.50）
    (0.00, 4.00),                        # gain_slope （先验 1.50）
    (0.00, 0.60),                        # da_floor   （先验 0.20）
    (0.05, 1.20),                        # da_slope   （先验 0.80）
]
PRIOR = np.array([0.70, 0.10, 0.50, 1.50, 0.20, 0.80])


def z_to_theta(z):
    """归一化坐标 -> 物理 θ 字典（含边界裁剪）。"""
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    v = lo + z * (hi - lo)
    return {k: float(x) for k, x in zip(THETA_KEYS, v)}


def theta_to_z(theta_vec):
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    return np.clip((np.asarray(theta_vec) - lo) / (hi - lo), 0.0, 1.0)


# ---------------- 手写 Jacobi 特征分解（规避 LAPACK 拦截） ----------------
def jacobi_eigh(A, max_iter=200, tol=1e-14):
    """对称小矩阵特征分解（Jacobi 旋转，纯 elementwise 运算）。
    返回 (evals 升序, evecs)，evecs 的第 i 列对应 evals[i]。"""
    n = A.shape[0]
    a = A.astype(float).copy()
    v = np.eye(n)
    for _ in range(max_iter):
        p, q, mx = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i, j]) > mx:
                    mx, p, q = abs(a[i, j]), i, j
        if mx < tol:
            break
        theta = 0.5 * np.arctan2(2.0 * a[p, q], a[q, q] - a[p, p])
        c, s = np.cos(theta), np.sin(theta)
        for k in range(n):
            akp, akq = a[k, p], a[k, q]
            a[k, p] = c * akp - s * akq
            a[k, q] = s * akp + c * akq
        for k in range(n):
            apk, aqk = a[p, k], a[q, k]
            a[p, k] = c * apk - s * aqk
            a[q, k] = s * apk + c * aqk
        for k in range(n):
            vkp, vkq = v[k, p], v[k, q]
            v[k, p] = c * vkp - s * vkq
            v[k, q] = s * vkp + c * vkq
    evals = np.diag(a).copy()
    order = np.argsort(evals)
    return evals[order], v[:, order]


def vnorm(x):
    """向量 2 范数（不用 np.linalg.norm，规避任何 LAPACK 路径）。"""
    return float(np.sqrt((np.asarray(x) ** 2).sum()))


# ---------------- CMA-ES（Hansen 标准形式，6 维小规模） ----------------
class CMAES:
    def __init__(self, x0, sigma, seed=0, popsize=None):
        self.n = len(x0)
        self.mean = np.asarray(x0, dtype=float)
        self.sigma = float(sigma)
        self.lam = popsize or (4 + int(3 * np.log(self.n)))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / float((self.weights ** 2).sum())
        n = self.n
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.inv_sqrt_C = np.eye(n)
        self.eigeneval = 0
        self.counteval = 0
        self.rng = np.random.default_rng(seed)

    def _update_eig(self):
        if self.counteval - self.eigeneval < self.lam / (self.c1 + self.cmu) / self.n / 10:
            return
        self.eigeneval = self.counteval
        C_sym = (self.C + self.C.T) / 2.0
        evals, self.B = jacobi_eigh(C_sym)
        self.D = np.sqrt(np.maximum(evals, 1e-20))
        # B diag(1/D) B^T（用 einsum：本机 GEMM 被拦截，GEMV 与 einsum 安全）
        self.inv_sqrt_C = np.einsum('ij,kj->ik', self.B / self.D, self.B)

    def ask(self):
        self._update_eig()
        pop = []
        for _ in range(self.lam):
            z = self.rng.standard_normal(self.n)
            y = self.B @ (self.D * z)
            pop.append(self.mean + self.sigma * y)
        return pop

    def tell(self, pop, fitness):
        order = np.argsort(fitness)
        pop = [pop[i] for i in order]
        ys = [(pop[i] - self.mean) / self.sigma for i in range(self.lam)]
        yw = np.zeros(self.n)
        for i in range(self.mu):
            yw += self.weights[i] * ys[i]
        self.mean = self.mean + self.sigma * yw
        self.counteval += self.lam
        # 步长控制
        self.ps = (1 - self.cs) * self.ps \
            + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.inv_sqrt_C @ yw)
        ps_norm = vnorm(self.ps)
        denom = np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lam)) * self.chiN
        hsig = (ps_norm / max(denom, 1e-12)) < (1.4 + 2 / (self.n + 1))
        # 秩-μ 与秩-1 协方差更新
        self.pc = (1 - self.cc) * self.pc \
            + (1.0 if hsig else 0.0) * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        artmp = np.array(ys[:self.mu])                       # (mu, n)
        # Σ w_i y_i y_iᵀ（einsum，规避 GEMM 拦截）
        C_mu = np.einsum('ki,kj->ij', artmp * self.weights[:, None], artmp)
        self.C = (1 - self.c1 - self.cmu) * self.C \
            + self.c1 * (np.outer(self.pc, self.pc)
                         + (0.0 if hsig else self.cc * (2 - self.cc)) * self.C) \
            + self.cmu * C_mu
        self.sigma *= float(np.exp((self.cs / self.damps) * (ps_norm / self.chiN - 1)))
        return order[0]                                      # 本代最优索引


# ---------------- 内环评估 ----------------
def eval_deterministic_full(graph, readout, safety, brain_mod):
    """确定性评估（无噪声、W 冻结）：六状态末端误差均值 + 平滑度（平均 |Δdq|）。"""
    errs, jerks = [], []
    for st in STATE_NAMES:
        brain = FlyBrain(graph, seed=42, mod=brain_mod)
        brain.reset()
        q = SIM_ZERO.copy()
        qt = pose_base(st)
        gamma = STATES[st]["gamma"]
        prev_dq = np.zeros(5)
        j = []
        for _ in range(int(4.0 * CONTROL_HZ)):
            x = build_sensory_input(st, q, qt, N_SENSORY)
            s = None
            for _ in range(SNN_STEPS_PER_CONTROL):
                s = brain.step(x, gamma)
            dq = readout.act(brain.dn_rates(s))
            q = safety.rate_limit(q + dq, q)
            j.append(float(np.abs(dq - prev_dq).mean()))
            prev_dq = dq
        errs.append(float(np.max(np.abs(q - qt))))
        jerks.append(float(np.mean(j)))
    return float(np.mean(errs)), float(np.mean(jerks))


def evaluate_theta(z, args):
    """单个 θ 的完整适应度评估：seeds × 内环训练 + 确定性评估。"""
    th = z_to_theta(z)
    brain_mod = {k: th[k] for k in ("leak_base", "leak_amp", "gain_bias", "gain_slope")}

    # 防作弊约束
    penalty = 0.0
    if th["da_floor"] + th["da_slope"] < 0.35:
        penalty += 1.0
    if th["gain_slope"] < 0.05:
        penalty += 0.5

    err_list, jerk_list, auc_list, curves = [], [], [], []
    for si in range(args.seeds):
        sd = args.seed + 1000 * si
        graph = ConnectomeGraph.build(seed=sd, variant="mock")
        brain = FlyBrain(graph, seed=sd, mod=brain_mod)
        readout = DNReadout(n_dn=N_DN, n_joints=5, seed=sd, trainable=True)
        readout.da_floor = th["da_floor"]
        readout.da_slope = th["da_slope"]
        safety = SafetyLayer()
        rng = np.random.default_rng(sd + 7)

        rewards = []
        for ep in range(args.episodes):
            st = STATE_NAMES[rng.integers(0, len(STATE_NAMES))]
            noise_std = max(0.1, 0.6 * (0.995 ** ep))
            r = run_episode(brain, readout, safety, st, rng, noise_std, "three_factor")
            rewards.append(r)
        curves.append(rewards)

        err, jerk = eval_deterministic_full(graph, readout, safety, brain_mod)
        err_list.append(err)
        jerk_list.append(jerk)

        k = max(1, args.episodes // 4)
        auc_list.append(float(np.mean(rewards[-k:]) - np.mean(rewards[:k])))

    err_mean = float(np.mean(err_list))
    err_std = float(np.std(err_list))
    jerk = float(np.mean(jerk_list))
    auc_gain = float(np.mean(auc_list))
    fitness = 1.0 * err_mean + 0.5 * err_std - 1.0 * auc_gain + 2.0 * jerk + penalty

    return {
        "theta": th, "fitness": float(fitness),
        "err_mean": err_mean, "err_std": err_std,
        "jerk": jerk, "auc_gain": auc_gain, "penalty": penalty,
        "curves": curves,
    }


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--popsize", type=int, default=6)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=40,
                    help="每个 θ、每个种子的内环训练 episode 数")
    ap.add_argument("--seeds", type=int, default=2, help="每个 θ 的评估种子数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=float, default=0.20, help="CMA-ES 初始步长（归一化空间）")
    args = ap.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    jsonl_path = os.path.join(data_dir, "evolve_modulation.jsonl")
    best_path = os.path.join(data_dir, "modulation_theta_best.json")

    x0 = theta_to_z(PRIOR)
    es = CMAES(x0, sigma=args.sigma, seed=args.seed, popsize=args.popsize)
    best = {"fitness": np.inf}
    t_start = time.time()

    print(f"[evo] Level A 启动：popsize={es.lam} generations={args.generations} "
          f"episodes={args.episodes} seeds={args.seeds} 内环=three_factor")
    print(f"[evo] 先验 θ0 = {z_to_theta(x0)}")

    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for g in range(args.generations):
            pop = es.ask()
            fits = []
            for i, z in enumerate(pop):
                t0 = time.time()
                res = evaluate_theta(z, args)
                fits.append(res["fitness"])
                rec = {"gen": g, "idx": i, "theta": res["theta"],
                       "fitness": res["fitness"], "err_mean": res["err_mean"],
                       "err_std": res["err_std"], "jerk": res["jerk"],
                       "auc_gain": res["auc_gain"], "penalty": res["penalty"],
                       "seeds": args.seeds, "episodes": args.episodes,
                       "wall_time_s": round(time.time() - t0, 1)}
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jf.flush()
                # level 2：逐 episode 奖励曲线
                csv_path = os.path.join(data_dir, f"evolve_curves_g{g}_i{i}.csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                    w = csv.writer(cf)
                    w.writerow(["episode"] + [f"seed{s}" for s in range(args.seeds)])
                    for ep in range(args.episodes):
                        w.writerow([ep] + [f"{c[ep]:.4f}" for c in res["curves"]])
                if res["fitness"] < best["fitness"]:
                    best = dict(rec)
                    with open(best_path, "w", encoding="utf-8") as bf:
                        json.dump({"theta": res["theta"], "fitness": res["fitness"],
                                   "gen": g, "idx": i,
                                   "err_mean": res["err_mean"], "auc_gain": res["auc_gain"],
                                   "jerk": res["jerk"]}, bf, ensure_ascii=False, indent=2)
                print(f"[evo] g{g:02d} i{i}  fit={res['fitness']:+.4f} "
                      f"err={res['err_mean']:.3f} auc={res['auc_gain']:+.3f} "
                      f"jerk={res['jerk']:.4f} pen={res['penalty']:.1f} "
                      f"({time.time()-t0:.0f}s)")
            ib = es.tell(pop, fits)
            print(f"[evo] g{g:02d} 完成：best={min(fits):+.4f} mean={np.mean(fits):+.4f} "
                  f"sigma={es.sigma:.3f} 累计 {time.time()-t_start:.0f}s "
                  f"(本代最优 idx={ib})")

    print(f"[evo] 完成。最优 θ = {best.get('theta')}")
    print(f"[evo] 最优适应度 = {best.get('fitness'):+.4f}，已保存 {best_path}")
    print(f"[evo] 全部评估记录：{jsonl_path}")


if __name__ == "__main__":
    main()

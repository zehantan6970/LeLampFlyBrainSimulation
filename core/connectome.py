# -*- coding: utf-8 -*-
"""支柱一：固定连接拓扑（Fixed wiring topology）。

设计纪律（源自 FlyDoom，docs/nn/flydoom.md 2.1）：
- 邻接关系只用 COO 边表 + bincount 消息传递，禁止稠密 N*N；
- 解剖边集一旦生成永不变动；可塑的仅是边上的权值；
- 权重初始化 w = log1p(synapse_count)（突触计数为解剖学代理量，非突触效能）；
- 三种图变体供消融：mock（重尾度分布）/ er（Erdos-Renyi）/ rewired（度保持重连零模型）。
"""
import numpy as np


class ConnectomeGraph:
    """稀疏有向连接图：src/dst/weight 三个等长数组即全部拓扑。"""

    def __init__(self, n, src, dst, weight, dn_idx, sensory_idx, meta=None):
        self.n = int(n)
        self.src = np.asarray(src, dtype=np.int64)
        self.dst = np.asarray(dst, dtype=np.int64)
        self.weight = np.asarray(weight, dtype=np.float64)
        self.dn_idx = np.asarray(dn_idx, dtype=np.int64)
        self.sensory_idx = np.asarray(sensory_idx, dtype=np.int64)
        self.meta = meta or {}

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, n=500, avg_degree=8, n_dn=10, n_sensory=24, seed=42, variant="mock"):
        """生成图。variant: mock | er | rewired（rewired = 对 mock 做度保持重连）"""
        rng = np.random.default_rng(seed)

        if variant == "er":
            m_target = n * avg_degree
            src = rng.integers(0, n, size=m_target)
            dst = rng.integers(0, n, size=m_target)
        else:
            # mock：对数正态出度（近似 MaleCNS 的重尾度分布统计量）
            out_deg = np.maximum(1, rng.lognormal(mean=np.log(avg_degree) - 0.5, sigma=1.0, size=n).astype(int))
            src = np.repeat(np.arange(n), out_deg)
            dst = rng.integers(0, n, size=src.shape[0])

        # 去自环、去重边
        mask = src != dst
        src, dst = src[mask], dst[mask]
        pair = src * n + dst
        pair = np.unique(pair)
        src, dst = pair // n, pair % n

        if variant == "rewired":
            src, dst = cls._degree_preserving_rewire(src, dst, n_swaps=len(src), rng=rng)

        # 权重：log1p(突触计数)，突触计数 ~ 1 + Poisson(3)
        synapse_count = 1 + rng.poisson(3, size=len(src))
        weight = np.log1p(synapse_count).astype(np.float64)

        # DN / 感觉节点：由种子固定，之后永不变动。
        # 关键设计（2026-09-28 实证修正）：感觉节点中前 12 个必须是 DN 的
        # 直接突触前驱（pre-DN）——通道 0-4 ON 误差、5-9 OFF 误差、10-11 备用。
        # 多跳衰减（0.25 增益）下误差信号到 DN 已无信噪比，必须单跳直达。
        perm = rng.permutation(n)
        dn_idx = perm[:n_dn]
        pre_dn = np.unique(src[np.isin(dst, dn_idx)])
        pre_dn = pre_dn[~np.isin(pre_dn, dn_idx)]
        rng.shuffle(pre_dn)
        n_pre = min(12, len(pre_dn))
        rest = perm[n_dn:][~np.isin(perm[n_dn:], pre_dn[:n_pre])]
        sensory_idx = np.concatenate([pre_dn[:n_pre], rest[:n_sensory - n_pre]])

        meta = {"variant": variant, "seed": seed, "n_edges": int(len(src)),
                "n_pre_dn_sensory": int(n_pre)}
        g = cls(n, src, dst, weight, dn_idx, sensory_idx, meta)
        # 默认按入强度归一化（设计文档 4.1 可选项）：
        # 否则重尾度分布下高入度节点接收 msg ~ in_degree * w >> 1，
        # tanh 全体饱和于 1，网络丧失状态可分性（2026-09-28 诊断实证）。
        g.in_strength_normalize()
        # 子临界缩放：归一化后 leak(0.7)+msg(~h) 的环路总增益 >1 会双稳态饱和；
        # 缩放至 0.25 使 leak+w_scale < 1，单稳态，感觉输入 x 才能主导状态分化。
        g.weight *= 0.25
        g.meta["recurrent_gain"] = 0.25
        # 感觉上行通路增益标定：入强度归一化会把 pre-DN 边稀释至 ~0.25/in_degree，
        # 误差通道对 DN 的调制被淹没（2026-09-28 实证：DN 对 ±3rad 误差无响应）。
        # 对 pre-DN 上行边做 4x 增益（仅缩放既有边权，不增删边，符合拓扑固定纪律）。
        boost = np.isin(g.src, g.sensory_idx[:10]) & np.isin(g.dst, g.dn_idx)
        g.weight[boost] *= 16.0
        g.meta["sensory_boost"] = 16.0
        return g

    # ------------------------------------------------------------------
    @staticmethod
    def _degree_preserving_rewire(src, dst, n_swaps, rng):
        """度保持双边交换零模型（Maslov & Sneppen, 2002）。"""
        src, dst = src.copy(), dst.copy()
        edge_set = set(zip(src.tolist(), dst.tolist()))
        m = len(src)
        done = 0
        attempts = 0
        while done < n_swaps and attempts < n_swaps * 10:
            attempts += 1
            i, j = rng.integers(0, m, size=2)
            a, b, c, d = src[i], dst[i], src[j], dst[j]
            if a == d or c == b or a == c or b == d:
                continue
            if (a, d) in edge_set or (c, b) in edge_set:
                continue
            edge_set.discard((a, b)); edge_set.discard((c, d))
            edge_set.add((a, d)); edge_set.add((c, b))
            dst[i], dst[j] = d, b
            done += 1
        return src, dst

    # ------------------------------------------------------------------
    def message_passing(self, h):
        """稀疏消息传递：msg[j] = sum_i w_ij * h[i]（edge gather + index_add）。"""
        contrib = h[self.src] * self.weight
        return np.bincount(self.dst, weights=contrib, minlength=self.n)

    def in_strength_normalize(self):
        """可选：按入强度归一化（设计文档 4.1 可选项）。"""
        in_str = np.bincount(self.dst, weights=self.weight, minlength=self.n)
        in_str[in_str == 0] = 1.0
        self.weight = self.weight / in_str[self.dst]

# LeLamp-FlyBrain Phase 1 — Simulation Demo Video Description

> 用途：GitHub 仓库 README / 录屏视频简介 / 视频字幕脚本。
> 录制命令（本工程根目录 PowerShell）：
>
> ```powershell
> .\.venv312\python.exe main.py --mode demo --controller brain --ckpt data\w_dn_calibrated.npz --engine mujoco
> ```

---

## 一句话摘要（可置于视频简介开头）

**EN**: A 5-DOF desk lamp driven entirely by a fixed-topology, Drosophila-inspired spiking neural network — six emotional behaviors emerge from a 10-neuron descending readout, modulated online by a γ=[DA, OA, 5-HT] neuromodulatory vector. No backpropagation, no learned connectivity, no dataset.

**中**：一盏由固定拓扑果蝇式脉冲神经网络全权驱动的 5 自由度台灯——六种情绪行为全部经由 10 个下降神经元读出产生，由 γ=[多巴胺, 章鱼胺, 血清素] 调控向量在线调制。无反向传播、无可学习连接、零数据集。

---

## 视频时间轴解说（约 38 秒，逐段对应）

| 时间 | 状态 | 画面中应看到 | γ 调制解读 | 控制台字幕 |
|---|---|---|---|---|
| 00:00–00:06 | **Vigilance_Defense 警戒** | 灯体立直，底座以 0.15 Hz 缓慢环视（±0.8 rad）；灯头冰蓝常亮 | 高 5-HT → 膜泄漏时间常数增大，网络迟缓持重，如警戒中的果蝇缓慢扫视 | "Scanning surroundings..." |
| 00:06–00:12 | **Approach_Reward 喜悦** | 灯体前倾，0.8 Hz 欢快摆头（±0.2 rad）；暖黄常亮 | 高 DA → 三因子学习门控全开，此时网络学得最快 | "Glad to be with you." |
| 00:12–00:18 | **Fear_Arousal 惊恐** | 灯体后撤高抬，全身 8 Hz 微颤；红色 10 Hz 爆闪 | 高 OA → 感觉注入增益放大至 1.85 倍（对误差最敏感）；低 5-HT → 反应敏捷 | "Watch out! Obstacle ahead." |
| 00:18–00:24 | **Exploration_Seeking 探索** | 低头巡航，0.4 Hz S 型弧线扫描；绿色交替呼吸 | 三个调制旋钮均居中——无偏置的中性探索态 | "Exploring your workspace." |
| 00:24–00:32 | **Emotional_Comfort 安抚** | 前倾下倾，0.3 Hz 呼吸律动；暖粉慢呼吸（3 s 周期） | 低 OA → 感觉输入被主动抑制，对外扰动不敏感，呈现"安稳"质感 | "I am here for you." |
| 00:32–00:38 | **Grooming_Rest 休眠** | 收拢垂首，完全静止；暗蓝 25% 亮度 | DA/OA 双压底 → 学习冻结 + 感觉抑制，进入休眠 | （无台词） |

---

## 画面中不可见、但建议在简介中说明的技术要点

1. **固定连接拓扑**：500 节点 mock 连接图（COO 稀疏），边集在全部 38 秒内**从未变动**——行为差异完全由同一拓扑的运行方式（γ 调制）产生，这是 FlyDoom 三支柱之一。
2. **唯一读出通路**：姿态 = `Δq = W · s_DN`，W 为 5×10 矩阵，仅允许从 10 个下降神经元读出。W 由岭回归标定（非反向传播），六状态确定性跟踪误差 **0.04 rad（≈2.3°）**，相对静止基线 0.542 rad 下降 92.6%。
3. **闭环伺服**：每个控制步（50 Hz），关节误差经 ON/OFF 整流感觉通道注入 SNN（100 Hz），网络对**运动参考**闭环跟踪——扫视、摆头、呼吸律动不是脚本回放，而是 SNN 实时伺服的结果。
4. **物理保真**：MuJoCo 引擎，kp=17.8 sts3215 舵机伺服模型，240 Hz 物理步进；单步增量限幅 0.06 rad（安全层）。
5. **事件流来源**：本演示使用本地 Mock 剧本（`utils/mock_scenarios.json`）。同一管线支持 DeepSeek 在线生成（`--mode online`），API 失败自动降级 Qwen → Mock，仿真不中断。

---

## 诚实声明（建议保留，体现研究严谨性）

- γ 三条通路（5-HT→时间常数、OA→感觉增益、DA→学习门控）为**工程类比**，非果蝇神经调质的生物学等效声明。
- 情绪姿态的**参考目标**（姿态基元）为手工设计的行为先验；本阶段验证的是「固定拓扑 + DN 读出」能否伺服复现这些行为，而非行为本身的涌现。
- 纯三因子生物学习（资格迹×多巴胺）当前在 T1 任务上尚不稳定（负结果已如实记录）；演示使用的 W 来自标定上界。学习规则改进为后续研究项。

---

## 录屏操作建议

1. **画面布局**：MuJoCo 窗口（主体）+ PowerShell 控制台（字幕）同屏录制；字幕只在控制台出现。
2. **分辨率**：窗口建议 1280×720 以上；控制台灯头颜色变化（尤其红色爆闪、暖粉呼吸）是视觉重点，勿裁剪灯头。
3. **一镜到底**：38 秒完整播放 + 结尾窗口保持画面 3–5 秒，展示「窗口保持打开」提示。
4. **可选第二段**：追加录一段 `--mode online --scenario "深夜用户加班疲惫，桌上有水杯"`，展示 LLM 现场生成中文事件流（约 10 秒生成 + 播放），与 Mock 段形成对比。
5. **发布格式**：MP4 (H.264)，≤60 s；仓库内同时保留本文件作为文字版说明。

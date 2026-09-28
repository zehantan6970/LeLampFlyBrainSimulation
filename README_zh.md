# LeLamp-FlyBrain 仿真系统

> 一盏由固定拓扑果蝇式脉冲神经网络全权驱动的 5 自由度台灯——六种情绪行为全部经由 10 个下降神经元（DN）读出产生，由 γ=[多巴胺, 章鱼胺, 血清素] 调控向量在线调制。**无反向传播、无可学习连接、零数据集。**

**English version (most detailed) → [README.md](README.md)**

![系统架构](docs/figures/fig1_architecture.png)

---

## 项目简介

**FLAMINGO / LeLamp-FlyBrain** 将两个方向结合：

1. **FlyDoom 式果蝇数字大脑**——500 节点固定稀疏循环网络（mock 连接图），边集在种子初始化后永不变动。行为由 **10 个下降神经元读出** `dq = W·s_DN` 产生，并由三分量神经调控向量 γ 在线调制（三支柱：固定拓扑、仅 DN 读出、三因子局部学习）。
2. **LLM 场景层**——DeepSeek → Qwen → 本地 Mock 三级熔断链，把自然语言场景解析为标准 JSON 时序事件流（携带情绪状态、γ 值、灯光模式与台词）。

最终效果：一台仿真的 LeLamp 5-DOF 台灯（MuJoCo，kp=17.8 STS3215 舵机保真模型）会扫视、摆头、震颤、呼吸、休眠——每个关节都由 **SNN 闭环伺服**（关节误差 ON/OFF 感觉通道 → SNN → DN 读出 → 安全层 → 执行器）。

**阶段一状态（2026-09-28）：三项验收标准全部达成**——全链路无头跑通；标定 DN 读出六状态平均跟踪误差 **0.040 rad**（较无控制基线 0.542 rad 下降 92.6%）；LLM 容错链在线（DeepSeek，约 10.5 s）与离线（Mock）均可用。

## 快速开始

```powershell
conda create -p .\.venv312 python=3.12 -y
.\.venv312\python.exe -m pip install numpy mujoco openai python-dotenv pillow

.\.venv312\python.exe main.py --mode demo                                            # GUI 演示（Mock，完全离线）
.\.venv312\python.exe experiments\calibrate_readout.py --samples 6000 --seed 42      # DN 读出标定
.\.venv312\python.exe main.py --mode demo --controller brain --ckpt data\w_dn_calibrated.npz --engine mujoco   # SNN 全权驱动
.\.venv312\python.exe main.py --mode online --scenario "深夜用户加班疲惫，桌上有水杯"  # LLM 在线生成
```

`.env` **不会被上传**（已 git-ignore）：复制 `.env.example` 为 `.env` 并填入自己的 DeepSeek / DashScope key；没有 key 时一切功能仍可通过本地 Mock 事件流运行。

详细教程（含实测输出与故障速查表）见 [RUN_GUIDE.md](RUN_GUIDE.md)；录屏解说稿见 [VIDEO_DESCRIPTION.md](VIDEO_DESCRIPTION.md)。

## 六种情绪状态

| 状态 | 姿态动作 | LED 灯光 | 台词 | γ=[DA,OA,5-HT] |
|------|---------|---------|------|----------------|
| Vigilance_Defense 警戒 | 立直 + 0.15 Hz 环视（±0.8 rad） | 冰蓝常亮 | "Scanning surroundings..." | [0.2, 0.5, 0.9] |
| Approach_Reward 喜悦 | 前倾 + 0.8 Hz 摆头（±0.2 rad） | 暖黄常亮 | "Glad to be with you." | [0.8, 0.2, 0.5] |
| Fear_Arousal 惊恐 | 后撤高抬 + 8 Hz 微颤 | 红色 10 Hz 爆闪 | "Watch out! Obstacle ahead." | [0.8, 0.9, 0.2] |
| Exploration_Seeking 探索 | 低头巡航 + 0.4 Hz S 型扫描 | 绿色交替呼吸 | "Exploring your workspace." | [0.5, 0.5, 0.4] |
| Emotional_Comfort 安抚 | 前倾下倾 + 0.3 Hz 呼吸律动 | 暖粉慢呼吸 | "I am here for you." | [0.3, 0.1, 0.9] |
| Grooming_Rest 休眠 | 收拢垂首，静止 | 暗蓝 25% 亮度 | （无台词） | [0.1, 0.1, 0.8] |

## 核心机制

![γ 三通路](docs/figures/fig2_gamma_pathways.png)

γ 由事件流逐事件下发，作用于 SNN 更新方程 `h ← tanh(leak·h + msg + x + b)`：**5-HT → 时间常数**（迟缓/敏捷）、**OA → 感觉注入增益**（唤醒）、**DA → 三因子学习门控**。γ 改变的是同一固定拓扑的「运行方式」，而非「连接结构」。*工程类比声明，非生物学等效。*

同一通路的符号化表达——系数 θ = (l₀, l₁, g₀, g₁, d₀, d₁) 即阶段二（层级 A）外环进化的优化目标（见 `experiments/evolve_modulation.py`）：

![γ 三通路（符号版）](docs/figures/fig5_gamma_symbolic.png)

![三支柱](docs/figures/fig3_three_pillars.png)

## 实测结果

![结果](docs/figures/fig4_results.png)

| 通道 | 机制 | T1 确定性评估 | 结论 |
|------|------|--------------|------|
| 岭回归标定 | 最小二乘（非生物） | **0.040 rad** | 架构上界（演示基准） |
| LMS 局部 delta 规则 | 生物可行，DA 门控 | 稳定但未达上界 | 当前最优学习路径 |
| 纯三因子（E×DA） | FlyDoom 核心主张 | 1.2–3.1 rad 摆动 | **负结果，如实记录** |

## 诚实声明

γ 三通路为工程类比；5-HT 通路在阶段一手工 θ 下近乎失效（leak ∈ [0.70, 0.75]），系数已可进化（P2-4 已实现 `experiments/evolve_modulation.py`，正式运行待定）；姿态目标为手工行为先验；mock 连接图非真实连接组；纯三因子学习未收敛（FlyDoom「负结果同样有效」立场）；gemini 早期文档 KPI 已降级为设计目标值。

## 参考与致谢

- [LeLamp](https://github.com/humancomputerlab/LeLamp)（GPL-3.0）与 lelamp_runtime——硬件构型与舵机参数；Apple [ELEGNT](https://arxiv.org/abs/2501.12493)（2025）——表达性运动方法论；[LeRobot](https://github.com/huggingface/lerobot)

> **引用说明**：若在学术工作中使用本项目，请在参考文献（References）中明确引用 LeLamp 官方 GitHub 仓库及 ELEGNT 论文。
- [FlyDoom](https://github.com/eganeganegan/flydoom)——三支柱框架直接来源；[awesome-fly](https://github.com/cobanov/awesome-fly)——全景调研
- FlyWire（*Nature* 2024）与 MaleCNS（*Cell* 2026-09）；[Shiu et al.](https://github.com/philshiu/Drosophila_brain_model) 与 [Eon Systems fly-brain](https://github.com/eonsystemspbc/fly-brain)；[flybody](https://github.com/TuragaLab/flybody) 与 [FlyGym](https://github.com/NeLy-EPFL/flygym)
- Frémaux & Gerstner (2016)——三因子学习规则综述
- MuJoCo（Todorov et al., 2012）与 PyBullet（Coumans & Bai）；DeepSeek API 与 Qwen（DashScope）；Google Gemini（早期设计草案辅助）

## 版权与开源协议声明

**本研究采用/改进了基于 GPL-3.0 协议开源的 LeLamp 硬件设计。**

本项目中的硬件设计基于 [LeLamp](https://github.com/humancomputerlab/LeLamp) 项目（上游项目）进行开发/衍生。根据 **GNU General Public License v3.0 (GPL-3.0)** 协议的要求：

1. **开源传染**：本项目的硬件资产（包括但不限于电路图、PCB 布线、3D 模型，以及衍生的仿真 URDF/MJCF 模型）同样以 **GPL-3.0** 协议完全开源。
2. **署名与免责**：本项目保留了 LeLamp 原作者的版权声明。本项目同样在「现状（As-Is）」基础上提供，不承担任何明示或暗示的担保责任。
3. **修改说明**：我们对原设计进行了以下修改：
   - 基座由浮动改为**固定基座**（仿真接地）；
   - 运动学树重构为**单串联链**（移除冗余固定关节）；
   - 采用**官方关节限位**与 **kp = 17.8 STS3215** 舵机参数；
   - STL 网格文件名 ASCII 化（原为非 ASCII 文件名）；
   - 新增 **MuJoCo MJCF 模型**（上游仅提供 URDF）；
   - 新增上游没有的控制栈：FlyDoom 式 SNN 控制器、γ 神经调控与 LLM 场景层。

> 本仓库为**纯软件仿真**，不分发 LeLamp 的 PCB/CAD 制造文件；`assets/` 中的仿真模型是依据上游硬件参数重建的。本仓库原创软件部分（SNN 控制器、仿真管线）的许可证将另行明确；注意硬件衍生资产的分发始终受 GPL-3.0 约束，整仓采用 GPL-3.0 是合规成本最低的选择。

**上游项目链接**：[LeLamp 官方仓库](https://github.com/humancomputerlab/LeLamp) · [ELEGNT 论文（Apple, 2025）](https://arxiv.org/abs/2501.12493)

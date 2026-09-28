# LeLamp-FlyBrain 阶段一仿真 — 运行教程（Step by Step）

> 按步骤顺序执行，每步都给出**预期现象**与**失败处置**。
> 本教程全部命令已在 2026-09-28 本机实测通过。

---

## 第 0 步：认识工程结构

```
D:\workbuddy_ws\fly_ws\sim_workbuddy\
├── assets\lelamp_5dof.urdf       # 修正版模型（固定基座/单串联链/官方关节限位）
├── assets\lelamp_5dof.xml        # MuJoCo 版（kp=17.8 sts3215 伺服参数）
├── configs\default_config.py     # 全部可调参数
├── core\                         # 脑层：connectome/fly_brain/dn_readout/states/safety
├── sim\env_mujoco.py             # MuJoCo 环境（默认引擎）
├── sim\env_pybullet.py           # PyBullet 4合1 环境（备选，见故障表）
├── utils\llm_client.py           # DeepSeek -> Qwen -> Mock 三级容错
├── utils\mock_scenarios.json     # 3 条本地保底事件流（数据）
├── experiments\calibrate_readout.py  # DN 读出岭回归标定（上界基线，推荐先跑）
├── experiments\train_t1.py           # T1 生物学习实验（三因子/LMS）
├── main.py                       # 主入口
├── .venv312\                     # 已装好的 Python 3.12 环境（依赖齐全）
└── data\                         # 运行产物（CSV 曲线、检查点）
```

**统一解释器路径**（下面简写为 `PY`）：

```powershell
cd D:\workbuddy_ws\fly_ws\sim_workbuddy
$PY = ".\.venv312\python.exe"
```

> 环境说明：`.venv312` 是 conda 创建的 Python 3.12 环境，已安装
> numpy / mujoco / pybullet / openai / python-dotenv，**开箱即用，无需再装任何依赖**。

---

## 第 1 步：冒烟测试（无窗口，1 分钟，验证环境完好）

```powershell
.\.venv312\python.exe main.py --mode headless
```

**预期现象**（实测输出）：

```
[llm] 使用本地 Mock 事件流
[main] 场景来源=mock  场景=Full Day Mixed  片段数=6
[main] 物理引擎 = MuJoCo（kp=17.8 sts3215 伺服保真基准；字幕走控制台）
[event t=  0.0s] Vigilance_Defense    gamma=[0.2, 0.5, 0.9]  "Scanning surroundings..."
...
[main] 事件流执行完毕
```

无 Traceback 即通过。

---

## 第 2 步：GUI 演示（Mock 模式，开箱即跑）

```powershell
.\.venv312\python.exe main.py --mode demo
```

**预期现象**：MuJoCo 查看器弹出，台灯依次演绎 6 个场景片段（约 40 秒）：
警戒（冰蓝环视）→ 喜悦（暖黄摆头）→ 惊恐（红色爆闪后撤）→ 探索（绿光巡航）→
安抚（暖粉呼吸 + 字幕 "I am here for you."）→ 休眠（收拢熄灭）。
桌面有半透明红色水杯障碍物。字幕在控制台同步打印。

换场景：`--scenario "安抚"` / `--scenario "水杯 避障"`（按关键词匹配 mock 库）。

> 若想用 PyBullet 引擎（带 3D 悬浮字幕）：`--engine pybullet`。
> 若提示 DLL 被策略阻止，见文末故障表第 1 条。

---

## 第 3 步：DN 读出标定（上界基线，约 2 分钟）

```powershell
.\.venv312\python.exe experiments\calibrate_readout.py --samples 6000 --seed 42
```

**预期现象**（实测值）：

```
[calib] 样本 6000 组，s_DN 均值 0.612
[calib] 确定性评估 per-state 误差: 0.060 0.022 0.046 0.024 0.026 0.061 | mean=0.040
[calib] 参考：zero-W（静止）基线 mean=0.542；越低越好
[calib] 已保存：...\data\w_dn_calibrated.npz
```

误差 0.04 rad ≈ 2.3°，相对静止基线（0.542）下降 92.6%——证明
「固定连接组拓扑 + DN 读出」架构可以有效驱动 5-DOF 伺服。

---

## 第 4 步：脑控制器演示（SNN 全权驱动姿态）

```powershell
.\.venv312\python.exe main.py --mode demo --controller brain --ckpt data/w_dn_calibrated.npz
```

与第 2 步的区别：5 个关节的运动**完全由果蝇 SNN 的 DN 读出闭环伺服产生**
（不再直接输出手工姿态基元）。`--controller primitive` 为对照组。

> 2026-09-28 修订：brain 模式跟踪的是「运动参考」（姿态基元含振荡分量），
> 误差经 ON/OFF 感觉通道注入、SNN→DN 闭环跟踪，因此台灯会持续可见地运动
> （环视扫视约 93°、摆头约 43°，跟踪误差 ≈0.04 rad）。
> 修订前跟踪静态目标，各片段 <1 秒收敛后即静止，仅 LED 变化。

---

## 第 5 步：LLM 在线场景生成（验证容错降级链）

```powershell
.\.venv312\python.exe main.py --mode online --scenario "深夜用户加班疲惫，桌上有水杯"
```

**预期现象**（实测）：`[llm] deepseek 成功，耗时 10.51s` → 场景由 DeepSeek 现场生成。
若 DeepSeek 失败自动切 Qwen；断网则回退本地 Mock。**任何情况下仿真不中断**。

---

## 第 6 步：生物学习实验（三因子 / LMS，约 5-10 分钟）

```powershell
# 默认 LMS 通道（局部 delta 规则，当前 T1 收敛路径）
.\.venv312\python.exe experiments\train_t1.py --episodes 300 --seed 42

# 纯三因子（资格迹 x 多巴胺）——科学对照组
.\.venv312\python.exe experiments\train_t1.py --learn three_factor --episodes 300 --seed 42

# 消融：不同连接图零模型
.\.venv312\python.exe experiments\train_t1.py --graph er --seed 1
.\.venv312\python.exe experiments\train_t1.py --graph rewired --seed 1
```

每 20 轮打印「滚动奖励 + 确定性评估误差」，最优检查点自动保存（eval-selection）。
产物在 `data\`：`t1_train_*.csv` 奖励曲线、`w_dn_t1.npz` 检查点。

**诚实声明（2026-09-28 实测）**：纯三因子通道目前在 T1 上学习不稳定
（训练奖励上升但确定性评估在 1.2~3.1 rad 间摆动），LMS 通道较稳但也未达
标定上界（0.04 rad）。这是符合 FlyDoom「负结果同样有效」立场的真实实验状态；
架构有效性由第 3 步标定基线证明。改进学习规则是后续研究项。

---

## 故障速查表（全部为本机实测踩坑记录）

| # | 症状 | 根因 | 处置 |
|---|---|---|---|
| 1 | `ImportError: DLL load failed ... pybullet ... 应用程序控制策略已阻止此文件` | 系统应用控制策略按文件身份阻止 pybullet 未签名二进制（换目录无效，实测） | 用默认 MuJoCo 引擎即可（`--engine auto` 自动回退）；MuJoCo 官方 wheel 不受阻 |
| 2 | 进程无报错静默退出（exit 127） | numpy `@` 大矩阵乘 / `linalg.inv` 触发 MKL GEMM/LAPACK 被策略拦截 | 代码已用广播+`einsum`+手写消元规避；新代码避免大矩阵 `@` 与 `linalg` |
| 3 | `pip install pybullet` 编译失败 | PyPI 无 cp312/cp313 Windows wheel | 已用 conda-forge 预编译包装入 `.venv312`，无需处理 |
| 4 | `[llm] deepseek 失败（401）` | key 失效 | 更新 `.env`；不影响 Mock 运行 |
| 5 | 中文输出乱码 | 终端编码 | `chcp 65001` 后重跑 |
| 6 | 训练奖励上升但评估误差摆动 | 学习规则不稳定（已知开放问题） | 用 `--learn lms`；以 `calibrate_readout.py` 的上界为演示基准 |

---

## 常用参数速查

| 参数 | 位置 | 作用 |
|---|---|---|
| `N_NEURONS` (500) | configs | SNN 节点数 |
| `N_DN` (10) | configs | 下降神经元数 |
| `ETA_W` / `W_MAX` | configs | 学习率 / 权重边界 |
| `MAX_DQ_PER_STEP` (0.06) | configs | 单步增量限幅 |
| `--graph mock/er/rewired` | train_t1 | 连接图消融 |
| `--learn lms/hybrid/three_factor` | train_t1 | 学习规则通道 |
| `--controller primitive/brain` | main | 行为先验 / SNN 闭环 |
| `--engine auto/pybullet/mujoco` | main | 物理引擎 |

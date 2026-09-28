# -*- coding: utf-8 -*-
"""FLAMINGO / LeLamp-FlyBrain 阶段一默认配置。

对应设计文档 docs/sim/phase1-simulation-design-workbuddy.md：
- F5 仿真零位约定：sim_zero = 各关节 range 中点
- F2 伺服参数以 sts3215 实际生效值为准（kp=17.8 语义仅作参考，PyBullet 用位置控制增益近似）
"""
import numpy as np

# ---------- 关节（与 assets/lelamp_5dof.urdf 完全一致） ----------
JOINT_NAMES = ["joint1_yaw", "joint2_pitch", "joint3_pitch", "joint4_roll", "joint5_pitch"]
JOINT_LOWER = np.array([-5.021034, -1.084258, -2.820024, -3.786540, -0.853337])
JOINT_UPPER = np.array([1.262151, 2.057335, 0.321569, 2.496645, 2.288255])
SIM_ZERO = (JOINT_LOWER + JOINT_UPPER) / 2.0          # F5：仿真零位约定 = range 中点
MAX_DQ_PER_STEP = 0.06                                 # L2 单步增量限幅（rad，50Hz 下约 3 rad/s）

# ---------- SNN / FlyBrain ----------
N_NEURONS = 500                                        # mock 连接图节点数（设计文档 4.1）
N_DN = 10                                              # 下降神经元数量（设计文档 4.2）
N_SENSORY = 24                                         # 感觉节点数（状态编码注入位点）
AVG_DEGREE = 8                                         # mock 图平均出度
LEAK_BASE = 0.70                                       # 膜电位泄漏基准（5-HT 在其上调制）
GAMMA_TRACE = 0.95                                     # 资格迹衰减
ETA_W = 0.01                                           # 三因子学习率
W_MAX = 0.5                                            # 权重边界（稳态，防饱和撞墙）
W_DECAY = 0.002                                        # 突触稳态衰减（每次更新向 0 收缩）
V_LR = 0.05                                            # 价值读出局部 delta 规则学习率
DISCOUNT = 0.97                                        # TD 折扣

# ---------- 控制与仿真 ----------
CONTROL_HZ = 50                                        # L2 输出频率
SNN_STEPS_PER_CONTROL = 2                              # 每个控制步 SNN 推进 2 步（SNN 100Hz）
PHYSICS_DT = 1.0 / 240.0
EPISODE_MAX_SECONDS = 60.0                             # 单 episode 上限（对应舵机 1 小时约束精神）

# ---------- T1 任务奖励（工程多巴胺类比，非生物模型） ----------
R_ALIVE = -0.01                                        # 每步存活成本（FlyDoom 惯例）
R_POSE_SCALE = 2.0                                     # 姿态误差惩罚系数：r = -scale * ||q - q_prim||
R_NEAR_BONUS = 1.0                                     # 进入基元邻域奖励
NEAR_THRESHOLD = 0.10                                  # 邻域阈值（rad，全关节最大偏差）

# ---------- LLM ----------
LLM_TIMEOUT_S = 3.0
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-max"

URDF_PATH = "assets/lelamp_5dof.urdf"
MOCK_SCENARIO_PATH = "utils/mock_scenarios.json"
CHECKPOINT_PATH = "data/w_dn_t1.npz"

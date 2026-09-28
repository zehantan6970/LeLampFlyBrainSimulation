# -*- coding: utf-8 -*-
"""FLAMINGO / LeLamp-FlyBrain 阶段一主入口。

用法：
    python main.py --mode demo      # 本地 Mock 事件流 + GUI 演示（默认，开箱即跑）
    python main.py --mode online --scenario "深夜用户很疲惫"   # LLM 在线生成（自动降级）
    python main.py --mode headless  # 无 GUI 冒烟测试（CI / 快速验证）

控制器：
    --controller primitive  姿态由行为先验（姿态基元）驱动，SNN 并行走通链路（M2 默认）
    --controller brain      姿态由 DN 读出驱动（需先运行 experiments/train_t1.py）
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.default_config import (  # noqa: E402
    CONTROL_HZ, SNN_STEPS_PER_CONTROL, SIM_ZERO, N_DN, CHECKPOINT_PATH,
)
from core.connectome import ConnectomeGraph  # noqa: E402
from core.fly_brain import FlyBrain  # noqa: E402
from core.dn_readout import DNReadout  # noqa: E402
from core.states import (  # noqa: E402
    STATES, state_onehot, pose_primitive, pose_base, led_rgba, build_sensory_input,
)
from core.safety import SafetyLayer  # noqa: E402
from utils.llm_client import generate_scenario  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def create_env(engine, gui, obstacles):
    """按引擎偏好创建仿真环境。auto：pybullet 可用则 pybullet，否则 mujoco。"""
    errors = []
    if engine in ("auto", "pybullet"):
        try:
            from sim.env_pybullet import LampEnv
            env = LampEnv(gui=gui, obstacles=obstacles, project_root=PROJECT_ROOT)
            print("[main] 物理引擎 = PyBullet（4合1 UI：姿态+LED+悬浮字幕+障碍物）")
            return env
        except Exception as e:  # noqa: BLE001
            errors.append(f"pybullet: {type(e).__name__}: {e}")
            if engine == "pybullet":
                raise
    if engine in ("auto", "mujoco"):
        try:
            from sim.env_mujoco import LampEnvMuJoCo
            env = LampEnvMuJoCo(gui=gui, obstacles=obstacles, project_root=PROJECT_ROOT)
            print("[main] 物理引擎 = MuJoCo（kp=17.8 sts3215 伺服保真基准；字幕走控制台）")
            return env
        except Exception as e:  # noqa: BLE001
            errors.append(f"mujoco: {type(e).__name__}: {e}")
    raise RuntimeError("无可用物理引擎：" + " | ".join(errors))


def build_brain(seed=42, graph_variant="mock", controller="primitive", ckpt=None):
    graph = ConnectomeGraph.build(seed=seed, variant=graph_variant)
    brain = FlyBrain(graph, seed=seed)
    readout = DNReadout(n_dn=N_DN, n_joints=5, seed=seed, trainable=False)
    if controller == "brain":
        ckpt_path = ckpt or os.path.join(PROJECT_ROOT, CHECKPOINT_PATH)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
        if not os.path.exists(ckpt_path):
            print(f"[main] 未找到检查点 {ckpt_path}")
            print("[main] 请先运行：python experiments/calibrate_readout.py "
                  "（或 experiments/train_t1.py）")
            print("[main] 本次回退为 primitive 控制器")
            controller = "primitive"
        else:
            readout.load(ckpt_path)
            print(f"[main] 已加载 DN 读出：{ckpt_path}")
    return graph, brain, readout, controller


def run(mode, scenario, seed, graph_variant, controller, max_seconds, engine, ckpt):
    data, source = generate_scenario(
        scenario or "mixed demo", PROJECT_ROOT,
        force_mock=(mode != "online"),
    )
    print(f"[main] 场景来源={source}  场景={data['scenario']}  片段数={len(data['timeline'])}")

    graph, brain, readout, controller = build_brain(seed, graph_variant, controller,
                                                    ckpt=ckpt)
    safety = SafetyLayer()

    obstacles_all = {}
    for ev in data["timeline"]:
        for ob in ev.get("environmental_obstacles", []):
            obstacles_all[ob["id"]] = ob
    env = create_env(engine, gui=(mode != "headless"),
                     obstacles=list(obstacles_all.values()))

    q = SIM_ZERO.copy()
    hz = CONTROL_HZ
    brain.reset()
    t0 = time.time()

    try:
        for ev in data["timeline"]:
            state_name = ev["fruit_fly_state"]
            st = STATES[state_name]
            nm = ev.get("neuromodulation", {})
            gamma = [nm.get("dopamine_DA", 0.5), nm.get("octopamine_OA", 0.2),
                     nm.get("serotonin_5HT", 0.5)]
            text = ev.get("output_response", {}).get("speak_text", st["text"])
            dur = float(ev.get("duration_s", 6.0))
            if max_seconds and (time.time() - t0) > max_seconds:
                break

            print(f"[event t={ev.get('timestamp', 0):>5.1f}s] {state_name:<20s} "
                  f"gamma={gamma}  \"{text}\"")

            env.set_text(text)
            x = state_onehot(state_name, len(graph.sensory_idx))
            n_steps = int(dur * hz)
            prim_start = time.time()

            for k in range(n_steps):
                t_rel = k / hz                       # 片段内时间（驱动振荡）
                # --- L1 脑层推进（SNN 100Hz） ---
                if controller == "brain":
                    # 跟踪"运动参考"：姿态基元的振荡分量作为外部移动目标，
                    # 误差经 ON/OFF 感觉通道注入，SNN→DN 闭环跟踪。
                    # 2026-09-28 修订：原静态 pose_base 使各片段 <1s 收敛后
                    # 全程静止，演示不可感知；运动参考不改变"SNN 全权驱动"性质。
                    qt = pose_primitive(state_name, t_rel)
                    x = build_sensory_input(state_name, q, qt,
                                            len(graph.sensory_idx))
                s = None
                for _ in range(SNN_STEPS_PER_CONTROL):
                    s = brain.step(x, gamma)
                s_dn = brain.dn_rates(s)

                # --- 控制律 ---
                if controller == "brain":
                    dq = readout.act(s_dn)
                    q_cmd = safety.rate_limit(q + dq, q)
                else:  # primitive：行为先验驱动，经同一安全层
                    q_cmd = safety.rate_limit(pose_primitive(state_name, t_rel), q)

                env.set_joint_targets(q_cmd)
                env.set_led(led_rgba(state_name, time.time() - prim_start))
                env.step(n=int((1.0 / hz) / (1.0 / 240.0)))
                q = q_cmd

                # 碰撞风险监视（头端球-AABB）
                if (k % 25) == 0 and env.obstacles:
                    hit, clear = safety.head_collision_risk(env.get_head_position(),
                                                            env.obstacles)
                    if hit:
                        print(f"  [safety] 头端接近障碍物！净距离 {clear*100:.1f}cm")

        print("[main] 事件流执行完毕")
    finally:
        if mode == "headless":
            env.close()
        else:
            print("[main] 窗口保持打开，关闭窗口或按 Ctrl+C 退出")
            try:
                while True:
                    env.step(n=24)
            except (KeyboardInterrupt, Exception):  # noqa: BLE001
                env.close()


def main():
    ap = argparse.ArgumentParser(description="LeLamp-FlyBrain Phase-1 Simulation")
    ap.add_argument("--mode", choices=["demo", "online", "headless"], default="demo")
    ap.add_argument("--scenario", type=str, default="mixed demo")
    ap.add_argument("--controller", choices=["primitive", "brain"], default="primitive")
    ap.add_argument("--graph", choices=["mock", "er", "rewired"], default="mock")
    ap.add_argument("--engine", choices=["auto", "pybullet", "mujoco"], default="auto")
    ap.add_argument("--ckpt", type=str, default="",
                    help="brain 控制器的读出权重路径（默认 data/w_dn_t1.npz）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help=">0 时限制总运行时长（headless 冒烟用）")
    args = ap.parse_args()
    if args.mode == "headless" and args.max_seconds <= 0:
        args.max_seconds = 8.0
    run(args.mode, args.scenario, args.seed, args.graph, args.controller,
        args.max_seconds, args.engine, args.ckpt or None)


if __name__ == "__main__":
    main()

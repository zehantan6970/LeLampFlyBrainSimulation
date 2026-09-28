# -*- coding: utf-8 -*-
"""L3 MuJoCo 环境（主用引擎：动力学保真基准，kp=17.8 sts3215 伺服模型）。

与 LampEnv(env_pybullet) 提供相同接口，供 main.py 无差别调用。
GUI 使用官方 passive viewer；headless 模式纯数值推进。
LED 通过运行时改写 head_geom 的 rgba 实现；字幕输出到控制台（MuJoCo 无 3D 文字 API）。
"""
import os
import tempfile
import time

import numpy as np
import mujoco
import mujoco.viewer

from configs.default_config import JOINT_NAMES, SIM_ZERO, PHYSICS_DT

_XML_NAME = "lelamp_5dof.xml"


class LampEnvMuJoCo:
    def __init__(self, gui=True, obstacles=None, project_root=None):
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.gui = gui
        self.obstacles = obstacles or []

        xml_path = os.path.join(self.project_root, "assets", _XML_NAME)
        with open(xml_path, "r", encoding="utf-8") as f:
            xml = f.read()

        # 注入障碍物体素（半透明红盒）
        ob_xml = ""
        for ob in self.obstacles:
            hx, hy, hz = [s / 2.0 for s in ob["size"]]
            px, py, pz = ob["pos"]
            ob_xml += (f'<geom name="ob_{ob["id"]}" type="box" '
                       f'size="{hx} {hy} {hz}" pos="{px} {py} {pz}" '
                       f'rgba="1 0 0 0.35" contype="0" conaffinity="0"/>\n')
        xml = xml.replace("<!-- OBSTACLES_PLACEHOLDER -->", ob_xml)

        # 写入临时文件加载（避免改动资产原件）
        self._tmp = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                                encoding="utf-8")
        self._tmp.write(xml)
        self._tmp.close()
        self.model = mujoco.MjModel.from_xml_path(self._tmp.name)
        self.data = mujoco.MjData(self.model)

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in JOINT_NAMES]
        self.qpos_adr = [self.model.jnt_qposadr[j] for j in self.joint_ids]
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                          f"act_j{i+1}") for i in range(5)]
        self.head_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                              "head_geom")
        self.head_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                              "head_tip")

        self.q = SIM_ZERO.copy()
        self.reset_pose()
        mujoco.mj_forward(self.model, self.data)

        self.viewer = None
        if gui:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # ------------------------------------------------------------------
    def reset_pose(self):
        for adr, v in zip(self.qpos_adr, SIM_ZERO):
            self.data.qpos[adr] = v
        self.set_joint_targets(SIM_ZERO)

    def set_joint_targets(self, q):
        q = np.asarray(q, dtype=float)
        for act, v in zip(self.act_ids, q):
            self.data.ctrl[act] = v
        self.q = q.copy()

    def get_head_position(self):
        return np.asarray(self.data.site_xpos[self.head_site_id]).copy()

    def set_led(self, rgba):
        self.model.geom_rgba[self.head_geom_id] = rgba

    def set_text(self, text, rgb=None):
        if text:
            print(f'  [caption] "{text}"')

    def step(self, n=1):
        steps = max(1, int(n * PHYSICS_DT / self.model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()
                time.sleep(n * PHYSICS_DT * 0.9)
            else:
                raise KeyboardInterrupt("viewer closed")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

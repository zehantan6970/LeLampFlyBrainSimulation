# -*- coding: utf-8 -*-
"""L3 PyBullet 4 合 1 可视化环境：姿态 + LED + 3D 悬浮字幕 + 障碍物体素。"""
import os
import time

import numpy as np
import pybullet as p
import pybullet_data

from configs.default_config import (
    URDF_PATH, JOINT_NAMES, SIM_ZERO, PHYSICS_DT,
)


class LampEnv:
    def __init__(self, gui=True, obstacles=None, project_root=None):
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.gui = gui
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(PHYSICS_DT)

        p.loadURDF("plane.urdf")

        urdf = os.path.join(self.project_root, URDF_PATH)
        self.robot = p.loadURDF(urdf, basePosition=[0, 0, 0], useFixedBase=True)

        # 建立 joint 名 -> 索引映射（禁止依赖 link/joint 名推断功能以外的假设）
        self.joint_indices = []
        name2idx = {}
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            name2idx[info[1].decode("utf-8")] = i
        for name in JOINT_NAMES:
            self.joint_indices.append(name2idx[name])
        self.head_link_index = self.joint_indices[-1]   # lamp_head 为 j5 的 child link

        self.q = SIM_ZERO.copy()
        self._apply(self.q, force=True)

        # 障碍物体素（半透明红）
        self.obstacles = obstacles or []
        for ob in self.obstacles:
            self._spawn_obstacle(ob)

        self._text_item = None
        self._cam_set = False

    # ------------------------------------------------------------------
    def _spawn_obstacle(self, ob):
        half = [s / 2.0 for s in ob["size"]]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[1, 0, 0, 0.35])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                          baseVisualShapeIndex=vis, basePosition=ob["pos"])

    def _apply(self, q, force=False):
        for idx, val in zip(self.joint_indices, q):
            if force:
                p.resetJointState(self.robot, idx, float(val))
            p.setJointMotorControl2(self.robot, idx, p.POSITION_CONTROL,
                                    targetPosition=float(val),
                                    force=3.35, positionGain=0.5, velocityGain=0.6)
        self.q = np.asarray(q, dtype=float).copy()

    # ------------------------------------------------------------------
    def set_joint_targets(self, q):
        self._apply(q)

    def get_head_position(self):
        state = p.getLinkState(self.robot, self.head_link_index)
        return np.asarray(state[0])

    def set_led(self, rgba):
        p.changeVisualShape(self.robot, self.head_link_index, rgbaColor=rgba)

    def set_text(self, text, rgb=(1, 1, 1)):
        if self._text_item is not None:
            p.removeUserDebugItem(self._text_item)
            self._text_item = None
        if text:
            pos = self.get_head_position() + np.array([0, 0, 0.12])   # 灯头上方 12cm
            self._text_item = p.addUserDebugText(text, pos, textColorRGB=rgb,
                                                 textSize=1.4)

    def step(self, n=1):
        for _ in range(n):
            p.stepSimulation()
        if self.gui:
            if not self._cam_set:
                p.resetDebugVisualizerCamera(0.9, 35, -25, [0, 0, 0.35])
                self._cam_set = True
            time.sleep(PHYSICS_DT * n * 0.9)

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(self.cid)

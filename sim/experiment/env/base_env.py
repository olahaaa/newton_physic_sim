"""L1: 通用 Newton 环境外壳 — 封装 viewer、model、state、CUDA Graph。"""

from typing import Optional
import numpy as np
import warp as wp
import newton

from controller.base_controller import BaseController
from controller.ik_controller import IKController


class BaseEnv:
    """封装 Newton 引擎基础设施，不包含机器人/衣物具体逻辑。"""

    def __init__(self, headless: bool = False) -> None:
        self.viewer = newton.viewer.ViewerGL(headless=headless)

        self._use_graph = True  # 子类可覆写

        self.model: Optional[newton.Model] = None
        self.state_0 = None
        self.state_1 = None

        self.sim_time: float = 0.0
        self.frame_dt: float = 0.0  # 由子类设置
        self.graph = None

        self.controller: Optional[BaseController] = None

    # ═══════════════════════════════════════════
    # 两阶段初始化
    # ═══════════════════════════════════════════

    def initialize_resources(self) -> None:
        """在 env 构造完成后调用，创建 controller。"""
        self.setup_controller()

    def setup_controller(self) -> None:
        """硬编码创建 IKController。"""
        self.controller = IKController()
        self.controller.initialize_resources(self)

    # ═══════════════════════════════════════════
    # CUDA Graph
    # ═══════════════════════════════════════════

    def capture(self) -> None:
        """捕获仿真循环为 CUDA Graph。"""
        if self._use_graph and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    # ═══════════════════════════════════════════
    # 仿真步进
    # ═══════════════════════════════════════════

    def step(self, target: np.ndarray) -> None:
        """单步仿真：IK 求解 → 物理步进。"""
        self.controller.compute(self, target)

        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def simulate(self) -> None:
        """物理步进 — 由子类实现。"""
        raise NotImplementedError

    # ═══════════════════════════════════════════
    # 渲染
    # ═══════════════════════════════════════════

    def render(self) -> None:
        """渲染当前帧到 viewer。"""
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


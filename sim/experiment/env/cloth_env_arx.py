"""L3: ARX-X5 机器人定制 — 手指 SDF、关节增益。"""

import numpy as np
import warp as wp
import newton

from .cloth_env_base import ClothEnvBase


class ClothEnvARX(ClothEnvBase):
    """ARX-X5 双臂衣物操控仿真环境。

    硬编码场景资产：
      - 2 个 ARX-X5 机械臂（X5A_v2.urdf）
      - 1 块布料（Female_T_Shirt.usd，Style3D 求解器）
      - 1 块面板（board.stl，刚体）
      - 地面平面
    """

    # ═══════════════════════════════════════════
    # 硬编码资产路径（相对于 sim/experiment/）
    # ═══════════════════════════════════════════

    URDF_PATH: str = "assets/robot/ARX-X5/X5A_v2.urdf"

    # ═══════════════════════════════════════════
    # 硬编码位姿参数
    # ═══════════════════════════════════════════

    LEFT_ARM_POS: tuple = (0.0, 0.5, 0.0)
    RIGHT_ARM_POS: tuple = (0.0, -0.5, 0.0)

    # 初始关节角（6 个臂关节）
    INIT_Q: list = [0.0, 1.5707963, 1.5707963, 0.0, 0.0, 0.0]

    TABLE_HEIGHT: float = 0.0

    def __init__(self, headless: bool = False) -> None:
        super().__init__(headless)

    # ═══════════════════════════════════════════
    # 抽象方法实现
    # ═══════════════════════════════════════════

    def _setup_viewer_camera(self) -> None:
        """设置相机到双臂工作空间上方。"""
        self.viewer.set_camera(
            pos=wp.vec3((1.5, 0.0, 0.75 + self.TABLE_HEIGHT)),
            pitch=-30,
            yaw=180,
        )

    def _load_urdf_asset(self, urdf_path: str, xform: wp.transform) -> None:
        """加载单个 ARX-X5 URDF 到独立 ModelBuilder：

        1. 加载 URDF
        2. 识别手指 body (link6/link7/link8) → 构建 SDF + HYDROELASTIC
        3. 非手指 shape → 凸包近似
        4. 配置关节增益、力矩限制、armature
        5. finalize 单臂模型 → _single_arm_models
        6. 合并到主 scene
        """
        robot_builder = newton.ModelBuilder()
        robot_builder.default_shape_cfg = self.urdf_shape_cfg
        robot_builder.rigid_contact_margin = 0.001  # 1mm

        robot_builder.add_urdf(
            urdf_path,
            xform=xform,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        robot_builder.default_shape_cfg = self.shape_cfg

        # 识别夹爪 body 索引
        def find_body(name: str) -> int:
            return next(
                i for i, lbl in enumerate(robot_builder.body_label)
                if lbl.endswith(f"/{name}")
            )

        finger_body_indices = {
            find_body("link6"),
            find_body("link7"),
            find_body("link8"),
        }
        non_finger_shape_indices = []
        for shape_idx, body_idx in enumerate(robot_builder.shape_body):
            if body_idx in finger_body_indices and robot_builder.shape_type[shape_idx] == newton.GeoType.MESH:
                mesh = robot_builder.shape_source[shape_idx]
                if mesh is not None and mesh.sdf is None:
                    shape_scale = np.asarray(robot_builder.shape_scale[shape_idx], dtype=np.float32)
                    if not np.allclose(shape_scale, 1.0):
                        mesh = mesh.copy(vertices=mesh.vertices * shape_scale, recompute_inertia=True)
                        robot_builder.shape_source[shape_idx] = mesh
                        robot_builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
                    mesh.build_sdf(
                        max_resolution=self.hydro_mesh_sdf_max_resolution,
                        narrow_band_range=self.shape_cfg.sdf_narrow_band_range,
                        margin=self.shape_cfg.gap if self.shape_cfg.gap is not None else 0.05,
                    )
                robot_builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
            elif body_idx not in finger_body_indices:
                robot_builder.shape_flags[shape_idx] &= ~newton.ShapeFlags.HYDROELASTIC
                non_finger_shape_indices.append(shape_idx)

        robot_builder.approximate_meshes(
            method="convex_hull",
            shape_indices=non_finger_shape_indices,
            keep_visual_shapes=True,
        )

        # 设置初始关节角
        init_q = self.INIT_Q
        robot_builder.joint_q[:8] = [*init_q, 0.01, 0.01]
        robot_builder.joint_target_pos[:8] = [*init_q, 1.0, 1.0]

        # 关节增益
        robot_builder.joint_target_ke[:8] = [500.0] * 8
        robot_builder.joint_target_kd[:8] = [50.0] * 8
        robot_builder.joint_effort_limit[:6] = [80.0] * 6
        robot_builder.joint_effort_limit[6:8] = [20.0] * 2
        robot_builder.joint_armature[:6] = [0.1] * 6
        robot_builder.joint_armature[6:8] = [0.5] * 2

        # finalize 单臂模型 → 供 IK 使用
        self._single_arm_models.append(robot_builder.finalize())

        # 合并到主 scene
        self.scene.add_builder(robot_builder)

    # ═══════════════════════════════════════════
    # 钩子覆写
    # ═══════════════════════════════════════════

    def _pre_finalize_scene(self) -> None:
        """添加地面平面。"""
        self.scene.default_shape_cfg.ke = 5.0e4
        self.scene.default_shape_cfg.kd = 5.0e2
        self.scene.add_ground_plane()

"""L2: 机械臂+衣物仿真基类 — 场景构建、双求解器步进。"""

from abc import abstractmethod
import warp as wp
import copy
from pxr import Usd
import newton
import newton.usd
from newton.solvers import style3d
from newton._src.geometry.utils import load_mesh

from .base_env import BaseEnv
from utils.env_utils import resolve_asset_path


class ClothEnvBase(BaseEnv):
    """构建双臂+衣物的完整仿真场景。

    场景资产在 _build_scene() 中硬编码加载（不使用配置文件）。
    子类必须实现 _load_urdf_asset() 和 _setup_viewer_camera()。
    """

    # ═══════════════════════════════════════════
    # 硬编码仿真参数
    # ═══════════════════════════════════════════

    FPS: int = 60
    SIM_SUBSTEPS: int = 15
    ITERATIONS: int = 5
    COLLIDE_SUBSTEPS: int = 1

    NUM_ROBOT: int = 2
    ENDEFFECTOR_ID: int = 6
    NUM_ARM_JOINTS: int = 6
    NUM_GRIPPER_JOINTS: int = 2

    # 硬编码资产路径（相对于 sim/experiment/）
    CLOTH_USD_PATH: str = "assets/cloth/Female_T_Shirt.usd"
    CLOTH_PRIM_PATH: str = "/Root/Female_T_Shirt/Root_Garment"
    BOARD_STL_PATH: str = "assets/board/board.stl"

    def __init__(self, headless: bool = False) -> None:
        super().__init__(headless)

        # 从类常量读取参数
        self.num_robot = self.NUM_ROBOT
        self.endeffector_id = self.ENDEFFECTOR_ID
        self.num_arm_joints = self.NUM_ARM_JOINTS
        self.num_gripper_joints = self.NUM_GRIPPER_JOINTS

        # 时间参数
        self.frame_dt = 1.0 / self.FPS
        self.sim_substeps = self.SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = self.ITERATIONS
        self.collide_substeps = self.COLLIDE_SUBSTEPS

        # 构建场景
        self._build_scene()

        # 设置 viewer
        self.viewer.set_model(self.model)
        self._setup_viewer_camera()

        # 刚性 control
        self.control = self.model.control()

        # FK 初始化 → 将初始关节角复制到 control targets
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        for j in range(self.model.joint_count):
            jq_start = int(self.model.joint_q_start.numpy()[j])
            jq_end = int(self.model.joint_q_start.numpy()[j + 1])
            jqd_start = int(self.model.joint_qd_start.numpy()[j])
            jqd_end = int(self.model.joint_qd_start.numpy()[j + 1])
            if self.model.joint_type.numpy()[j] == newton.JointType.FREE:
                assert jq_end - jq_start == 7 and jqd_end - jqd_start == 6
                wp.copy(
                    self.control.joint_target_pos, self.model.joint_q,
                    dest_offset=jqd_start, src_offset=jq_start, count=6,
                )
            elif jq_end - jq_start > 0:
                assert jq_end - jq_start == jqd_end - jqd_start
                wp.copy(
                    self.control.joint_target_pos, self.model.joint_q,
                    dest_offset=jqd_start, src_offset=jq_start, count=jq_end - jq_start,
                )

        # 碰撞管线
        sdf_hydroelastic_config = newton.geometry.HydroelasticSDF.Config(
            output_contact_surface=hasattr(self.viewer, "renderer"),
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_hydroelastic_config,
            soft_contact_margin=0.01,
        )
        self.contacts = self.collision_pipeline.contacts()

        self._post_build_scene()

        self.capture()  # 必须在所有 solver 创建之后

    # ═══════════════════════════════════════════
    # 抽象方法 — 子类必须实现
    # ═══════════════════════════════════════════

    @abstractmethod
    def _load_urdf_asset(self, urdf_path: str, xform: wp.transform) -> None:
        """加载单个机械臂 URDF，配置关节增益、手指 SDF 等。"""
        ...

    @abstractmethod
    def _setup_viewer_camera(self) -> None:
        """设置 viewer 相机位置和朝向。"""
        ...

    # ═══════════════════════════════════════════
    # 钩子方法 — 子类可选覆写
    # ═══════════════════════════════════════════

    def _pre_finalize_scene(self) -> None:
        """在 model.finalize() 前调用（如添加地面）。"""
        pass

    def _post_build_scene(self) -> None:
        """在 model 完全就绪后调用。"""
        pass

    # ═══════════════════════════════════════════
    # 场景构建
    # ═══════════════════════════════════════════

    def _build_scene(self) -> None:
        """完整场景构建流水线（资产硬编码）。

        1. ModelBuilder → 2. 注册属性 → 3. ShapeConfig → 4. 加载资产
        → 5. 钩子 → 6. finalize → 7. 求解器
        """
        self.scene = newton.ModelBuilder()

        # 注册求解器自定义属性
        newton.solvers.SolverMuJoCo.register_custom_attributes(self.scene)
        newton.solvers.SolverStyle3D.register_custom_attributes(self.scene)

        # 默认碰撞参数
        self.scene.default_shape_cfg.ke = 5.0e4
        self.scene.default_shape_cfg.kd = 5.0e2
        self.scene.default_shape_cfg.kf = 1.0e3
        self.scene.default_shape_cfg.mu = 0.25

        # 三种 ShapeConfig
        self.shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=1e11,
            sdf_max_resolution=64,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-0.01, 0.01),
            gap=0.01,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        self.mesh_shape_cfg = copy.deepcopy(self.shape_cfg)
        self.mesh_shape_cfg.sdf_max_resolution = None
        self.mesh_shape_cfg.sdf_target_voxel_size = None
        self.mesh_shape_cfg.sdf_narrow_band_range = (-0.1, 0.1)
        self.hydro_mesh_sdf_max_resolution = 64

        self.urdf_shape_cfg = copy.deepcopy(self.shape_cfg)
        self.urdf_shape_cfg.is_hydroelastic = False
        self.urdf_shape_cfg.sdf_max_resolution = None
        self.urdf_shape_cfg.sdf_target_voxel_size = None
        self.urdf_shape_cfg.sdf_narrow_band_range = (-0.1, 0.1)

        self._single_arm_models = []

        # 硬编码加载资产
        self._add_cloth_asset()
        self._add_board_asset()
        self._add_robot_assets()

        self._pre_finalize_scene()

        self.model = self.scene.finalize(requires_grad=False)
        self.model.soft_contact_ke = 1e4
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = 0.25

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # 刚体求解器
        self.robot_solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=500,
            nconmax=500,
            iterations=15,
            ls_iterations=100,
            impratio=1000.0,
        )

        # 布料求解器 (Style3D)
        self.cloth_solver = newton.solvers.SolverStyle3D(
            model=self.model,
            iterations=self.iterations,
        )
        self.cloth_solver._precompute(self.scene)
        self.cloth_solver.collision.radius = 3.5e-3

    def _add_cloth_asset(self) -> None:
        """硬编码加载布料资产（Female_T_Shirt.usd，Style3D）。"""
        usd_path = resolve_asset_path(self.CLOTH_USD_PATH)
        usd_stage = Usd.Stage.Open(str(usd_path))
        usd_prim = usd_stage.GetPrimAtPath(self.CLOTH_PRIM_PATH)

        garment_mesh, garment_mesh_uv_indices = newton.usd.get_mesh(
            usd_prim,
            load_uvs=True,
            preserve_facevarying_uvs=True,
            return_uv_indices=True,
        )
        garment_mesh_uv = garment_mesh.uvs * 1.0e-3

        style3d.add_cloth_mesh(
            self.scene,
            pos=wp.vec3(0.0, 0.0, 0.95),
            rot=wp.quat_from_axis_angle(axis=wp.vec3(1.0, 0.0, 0.0), angle=wp.pi / 2.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            panel_verts=garment_mesh_uv.tolist(),
            panel_indices=garment_mesh_uv_indices.tolist(),
            indices=garment_mesh.indices.tolist(),
            vertices=garment_mesh.vertices.tolist(),
            density=0.3,
            scale=1.0,
            particle_radius=5.0e-3,
            tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
            edge_aniso_ke=wp.vec3(2.0e-5, 1.0e-5, 5.0e-6),
        )

    def _add_board_asset(self) -> None:
        """硬编码加载面板资产（board.stl，静态刚体）。"""
        stl_path = resolve_asset_path(self.BOARD_STL_PATH)
        mesh_points, mesh_indices = load_mesh(str(stl_path))
        mesh = newton.Mesh(mesh_points, mesh_indices)
        mesh.build_sdf(max_resolution=self.hydro_mesh_sdf_max_resolution)

        body = -1  # 静态物体（world body）
        self.scene.add_shape_mesh(
            body=body,
            mesh=mesh,
            cfg=self.mesh_shape_cfg,
        )

    def _add_robot_assets(self) -> None:
        """硬编码加载双臂 URDF — 委托给子类实现的 _load_urdf_asset()。

        每个臂的位姿由子类常量（或此处覆写）决定。
        """
        arm_poses = self._get_arm_poses()
        urdf_path = resolve_asset_path(self._get_urdf_path())
        for pose in arm_poses:
            self._load_urdf_asset(str(urdf_path), pose)

    def _get_urdf_path(self) -> str:
        """子类可覆写以指定不同的 URDF 文件。"""
        return getattr(self, "URDF_PATH", "assets/robot/ARX-X5/X5A_v2.urdf")

    def _get_arm_poses(self) -> list[wp.transform]:
        """子类可覆写以指定不同的臂基座位姿。"""
        left_pos = getattr(self, "LEFT_ARM_POS", (0.0, 0.5, 0.0))
        right_pos = getattr(self, "RIGHT_ARM_POS", (0.0, -0.5, 0.0))
        return [
            wp.transform(wp.vec3(*left_pos), wp.quat_identity()),
            wp.transform(wp.vec3(*right_pos), wp.quat_identity()),
        ]

    # ═══════════════════════════════════════════
    # 仿真循环
    # ═══════════════════════════════════════════

    def simulate(self) -> None:
        """双求解器步进循环。

        for substep:
            1. rebuild cloth BVH
            2. clear forces
            3. viewer forces (mouse drag)
            4. collision detection (每 collide_substeps 步)
            5. rigid solver step (MuJoCo) — 临时隐藏粒子
            6. cloth solver step (Style3D)
            7. swap states
        """
        self.cloth_solver.rebuild_bvh(self.state_0)

        for step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.viewer.apply_forces(self.state_0)

            if step % self.collide_substeps == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)

            particle_count = self.model.particle_count

            self.model.particle_count = 0
            self.model.shape_contact_pair_count = 0
            self.robot_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.model.particle_count = particle_count

            if particle_count > 0:
                self.state_0.particle_f.zero_()

            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

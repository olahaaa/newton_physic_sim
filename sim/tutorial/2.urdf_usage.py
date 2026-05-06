import warp as wp
import newton
import os
import copy
import numpy as np  
from newton.geometry import HydroelasticSDF


class URDFExample:
    def __init__(self, viewer):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.sim_time = 0.0
        self.collision_substeps = 1

# region shape_cfg
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
# endregion

# 导入urdf
        builder = newton.ModelBuilder()
        builder.default_shape_cfg = self.urdf_shape_cfg
        builder.rigid_gap = 0.001
        urdf_path = os.path.join(os.path.dirname(__file__), "assets/robot/ARX-X5/X5A.urdf")

        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            enable_self_collisions=True,
            parse_visuals_as_colliders=False,
        )
        builder.default_shape_cfg = self.shape_cfg
        
# mesh优化
        def find_body(name):
            return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))

        # Set SDF collisions on panda hand and fingers for hydroelastic contact
        finger_body_indices = {
            find_body("link6"),
            find_body("link7"),
            find_body("link8"),
        }
        non_finger_shape_indices = []
        for shape_idx, body_idx in enumerate(builder.shape_body):
            if body_idx in finger_body_indices and builder.shape_type[shape_idx] == newton.GeoType.MESH:
                mesh = builder.shape_source[shape_idx]
                if mesh is not None and mesh.sdf is None:
                    shape_scale = np.asarray(builder.shape_scale[shape_idx], dtype=np.float32)
                    if not np.allclose(shape_scale, 1.0):
                        # Hydroelastic mesh SDFs must be scale-baked for non-unit shape scale.
                        mesh = mesh.copy(vertices=mesh.vertices * shape_scale, recompute_inertia=True)
                        builder.shape_source[shape_idx] = mesh
                        builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
                    mesh.build_sdf(
                        max_resolution=self.hydro_mesh_sdf_max_resolution,
                        narrow_band_range=self.shape_cfg.sdf_narrow_band_range,
                        margin=self.shape_cfg.gap,
                    )
                builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
            elif body_idx not in finger_body_indices:
                non_finger_shape_indices.append(shape_idx)

        # Convert non-finger shapes to convex hulls
        builder.approximate_meshes(
            method="convex_hull", shape_indices=non_finger_shape_indices, keep_visual_shapes=True
        )

# 设置joint各种属性
        init_q = [0.0, 1.5707963, 1.5707963, 0.0, 0.0, 0.0]
        builder.joint_q[0:8] = [*init_q, 0.01, 0.01]
        builder.joint_target_pos[:8] = [*init_q, 0.01, 0.01]

        builder.joint_target_ke[:8] = [500.0] * 8
        builder.joint_target_kd[:8] = [50.0] * 8
        builder.joint_effort_limit[:6] = [80.0] * 6
        builder.joint_effort_limit[6:8] = [20.0] * 2
        builder.joint_armature[:6] = [0.1] * 6
        builder.joint_armature[6:8] = [0.5] * 2

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        sdf_hydroelastic_config = HydroelasticSDF.Config(
            output_contact_surface=hasattr(viewer, "renderer"),
        )
        self.collison_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_hydroelastic_config,
        )
        self.contacts = self.collison_pipeline.contacts()

# solver
        self.solver = newton.solvers.SolverMuJoCo(
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

        self.control = self.model.control()
        self.viewer.set_model(self.model)
        self.capture()
        
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
    
    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        for i in range(self.sim_substeps):
            if i % self.collision_substeps == 0:
                self.collison_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

if __name__ == "__main__":
    viewer = newton.viewer.ViewerGL()
    example = URDFExample(viewer)
    while viewer.is_running():
        if not viewer.is_paused():
            example.step()
        example.render()
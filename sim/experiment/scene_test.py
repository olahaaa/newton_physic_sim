import os
import warp as wp
import newton
import newton.usd
from newton.solvers import style3d
from pxr import Usd
import copy
from newton._src.geometry.utils import load_mesh


class SceneExample:
    def __init__(self, viewer):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 5
        self.collide_substeps = 1
        self.viewer = viewer

        builder = newton.ModelBuilder()

        # default config
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        # builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.25

        # basic shapecfg 在此基础上修改
        self.shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=1e11,
            sdf_max_resolution=64,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-0.01, 0.01),
            # contact_margin=0.01, 这个参数被官方更新为了gap
            gap=0.01,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        # mesh shapecfg
        self.mesh_shape_cfg = copy.deepcopy(self.shape_cfg)
        self.mesh_shape_cfg.sdf_max_resolution = None
        self.mesh_shape_cfg.sdf_target_voxel_size = None
        self.mesh_shape_cfg.sdf_narrow_band_range = (-0.1, 0.1)
        self.hydro_mesh_sdf_max_resolution = 64
        # urdf shapecfg
        self.urdf_shape_cfg = copy.deepcopy(self.shape_cfg)
        self.urdf_shape_cfg.is_hydroelastic = False
        self.urdf_shape_cfg.sdf_max_resolution = None
        self.urdf_shape_cfg.sdf_target_voxel_size = None
        self.urdf_shape_cfg.sdf_narrow_band_range = (-0.1, 0.1)
        
        # add cloth
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        usd_state = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets/cloth/Female_T_Shirt.usd"))
        usd_prim_garment = usd_state.GetPrimAtPath("/Root/Female_T_Shirt/Root_Garment")

        garment_mesh, garment_mesh_uv_indices = newton.usd.get_mesh(
            usd_prim_garment,
            load_uvs=True,
            preserve_facevarying_uvs=True,
            return_uv_indices=True,
        )
        garment_mesh_uv = garment_mesh.uvs * 1.0e-3

        style3d.add_cloth_mesh(
            builder,
            pos=wp.vec3(0, 0, 0.95),
            rot=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
            vel=wp.vec3(0, 0, 0),
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

        # add board
        mesh_path = os.path.join(os.path.dirname(__file__), "assets/board/board.stl")
        mesh_points, mesh_indices = load_mesh(mesh_path)
        mesh = newton.Mesh(mesh_points, mesh_indices)

        # body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.9), wp.quat_identity()), mass=1.0)
        body = -1 #world
        mesh.build_sdf(max_resolution=64,)
        builder.add_shape_mesh(
            body=body,
            mesh=mesh,
            cfg = self.mesh_shape_cfg,
        )

        # add robot arm
        urdf_path = os.path.join(os.path.dirname(__file__), "assets/robot/ARX-X5/X5A_v2.urdf")
        builder.default_shape_cfg = self.urdf_shape_cfg
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(1.5, 0.0, 0.0), wp.quat_identity()),
            enable_self_collisions=True,
            parse_visuals_as_colliders=False,
        )
        init_q = [0.0, 1.5707963, 1.5707963, 0.0, 0.0, 0.0]
        builder.joint_q[0:8] = [*init_q, 0.01, 0.01]
        builder.joint_target_pos[:8] = [*init_q, 0.01, 0.01]

        builder.joint_target_ke[:8] = [500.0] * 8
        builder.joint_target_kd[:8] = [50.0] * 8
        builder.joint_effort_limit[:6] = [80.0] * 6
        builder.joint_effort_limit[6:8] = [20.0] * 2
        builder.joint_armature[:6] = [0.1] * 6
        builder.joint_armature[6:8] = [0.5] * 2

        # finalize
        builder.add_ground_plane()
        self.model = builder.finalize()

        self.model.soft_contact_ke = 1e4
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = 0.25

        # rigid_solver
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
        # cloth_solver
        self.cloth_solver = newton.solvers.SolverStyle3D(
                model=self.model,
                iterations=self.iterations,
            )
        self.cloth_solver._precompute(builder,)
        self.cloth_solver.collision.radius = 3.5e-3



        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # self.contacts = self.model.contacts()

        # collision pipeline
        sdf_hydroelastic_config = newton.geometry.HydroelasticSDF.Config(
            output_contact_surface=True
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_hydroelastic_config,
            soft_contact_margin=0.01,
        )
        self.contacts = self.collision_pipeline.contacts()




        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.0, -1.7, 1.4), 0.0, -270.0)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.viewer.apply_forces(self.state_0)

            if _step % self.collide_substeps == 0:
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

    def step(self):
        if self.graph is not None:
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
    example = SceneExample(viewer)
    while viewer.is_running():
        if not viewer.is_paused():
            example.step()
        example.render()
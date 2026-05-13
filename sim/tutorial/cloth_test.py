import os
import warp as wp
import newton
import newton.usd
from newton.solvers import style3d
from pxr import Usd

class ClothExample:
    def __init__(self, viewer):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 4
        self.viewer = viewer

        builder = newton.ModelBuilder()
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)

        usd_state = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets/cloth/Female_T_Shirt.usd"))
        usd_prim_garment = usd_state.GetPrimAtPath("/Root/Female_T_Shirt/Root_Garment")

        garment_mesh, garment_mesh_uv_indices = newton.Mesh.create_from_usd(
            usd_prim_garment,
            load_uvs=True,
            preserve_facevarying_uvs=True,
            return_uv_indices=True,
        )

        # garment_mesh, garment_mesh_uv_indices = newton.usd.get_mesh(
        #     usd_prim_garment,
        #     load_uvs=True,
        #     preserve_facevarying_uvs=True,
        #     return_uv_indices=True,
        # )
        garment_mesh_uv = garment_mesh.uvs * 1.0e-3

        style3d.add_cloth_mesh(
            builder,
            pos=wp.vec3(0, 0, 0),
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

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.model.soft_contact_radius = 0.2e-2
        self.model.soft_contact_margin = 0.35e-2
        self.model.soft_contact_ke = 1.0e1
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 0.2
        self.model.set_gravity((0, 0, -9.81))

        self.solver = newton.solvers.SolverStyle3D(
            self.model,
            self.iterations,
        )
        self.solver._precompute(builder,)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

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
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
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
    example = ClothExample(viewer)
    while viewer.is_running():
        if not viewer.is_paused():
            example.step()
        example.render()
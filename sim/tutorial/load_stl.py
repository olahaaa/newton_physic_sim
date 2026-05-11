import newton
import os
import warp as wp
from newton._src.geometry.utils import load_mesh


print(newton.__version__)

builder = newton.ModelBuilder()
builder.add_ground_plane()

# add model
mesh_shape_cfg = newton.ModelBuilder.ShapeConfig(
    kh=1e11,
    sdf_max_resolution=None,
    sdf_target_voxel_size = None,
    is_hydroelastic=True,
    sdf_narrow_band_range=(-0.1, 0.1),
    gap=0.01,
    mu_torsional=0.0,
    mu_rolling=0.0,
)

mesh_path = os.path.join(os.path.dirname(__file__), "assets/board/board.stl")
mesh_points, mesh_indices = load_mesh(mesh_path)
mesh = newton.Mesh(mesh_points, mesh_indices)

body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
mesh.build_sdf(max_resolution=64,)
builder.add_shape_mesh(
    body=body,
    mesh=mesh,
    cfg = mesh_shape_cfg,
)


model = builder.finalize()

state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()

solver = newton.solvers.SolverXPBD(model, iterations=10)

fps = 60
frame_dt = 1.0 / fps
sim_substeps = 10
sim_dt = frame_dt / sim_substeps

def simulate():
    global state_0, state_1
    for _ in range(sim_substeps):
        state_0.clear_forces()

        model.collide(state_0, contacts)

        solver.step(state_0, state_1, control, contacts, sim_dt)

        # swap states
        state_0, state_1 = state_1, state_0


if wp.get_device().is_cuda:
    with wp.ScopedCapture() as capture:
        simulate()
    graph = capture.graph
else:
    graph = None

viewer = newton.viewer.ViewerGL()
viewer.set_model(model)

sim_time = 0.0
while viewer.is_running():
    if graph:
        wp.capture_launch(graph)
    else:
        simulate()
    viewer.begin_frame(sim_time)
    viewer.log_state(state_0)
    viewer.log_contacts(contacts, state_0)
    viewer.end_frame()
    sim_time += frame_dt

viewer
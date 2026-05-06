import newton
import warp as wp
import newton.usd
import newton.examples
from pxr import Usd

print(newton.__version__)

builder = newton.ModelBuilder()
builder.add_ground_plane()

#region build scene

# Height from which to drop shapes
drop_z = 2.0

# SPHERE
sphere_pos = wp.vec3(0.0, -4.0, drop_z)
body_sphere = builder.add_body(
    xform=wp.transform(p=sphere_pos, q=wp.quat_identity()),
    label="sphere",  # Optional: human-readable identifier
)
builder.add_shape_sphere(body_sphere, radius=0.5)

# CAPSULE
capsule_pos = wp.vec3(0.0, -2.0, drop_z)
body_capsule = builder.add_body(xform=wp.transform(p=capsule_pos, q=wp.quat_identity()), label="capsule")
builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)

# CYLINDER
cylinder_pos = wp.vec3(0.0, 0.0, drop_z)
body_cylinder = builder.add_body(xform=wp.transform(p=cylinder_pos, q=wp.quat_identity()), label="cylinder")
builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.6)

# Multi-Shape Collider
multi_shape_pos = wp.vec3(0.0, 2.0, drop_z)
body_multi_shape = builder.add_body(xform=wp.transform(p=multi_shape_pos, q=wp.quat_identity()), label="multi_shape")

# Now attach both a sphere and a box to the multi-shape body
# body-local shape offsets, offset sphere in x so the body will topple over
sphere_offset = wp.vec3(0.1, 0.0, -0.3)
box_offset = wp.vec3(0.0, 0.0, 0.3)
builder.add_shape_sphere(body_multi_shape, wp.transform(p=sphere_offset, q=wp.quat_identity()), radius=0.25)
builder.add_shape_box(body_multi_shape, wp.transform(p=box_offset, q=wp.quat_identity()), hx=0.25, hy=0.25, hz=0.25)

# Load a mesh from a USD file
usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))

# Add the mesh as a rigid body
mesh_pos = wp.vec3(0.0, 4.0, drop_z - 0.5)
body_mesh = builder.add_body(xform=wp.transform(p=mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)), label="bunny")
builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

#endregion

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

# num_frames = 500
# sim_time = 0.0
# for _ in range(num_frames):
#     if graph:
#         wp.capture_launch(graph)
#     else:
#         simulate()
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
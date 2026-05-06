import warp as wp
import newton
import newton.ik as ik
import os
import zmq

class IKExample:
    def __init__(self, viewer):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.graph = None
        

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # urdf assets
        urdf_path = os.path.join(os.path.dirname(__file__), "assets/robot/ARX-X5/X5A.urdf")
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )

        self.model = builder.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contact = self.model.contacts()
        self.viewer.set_model(self.model)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0) #更新state.body_q

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

# region ik setup
        # ee
        self.ee_index = 6
        body_q_np = self.state_0.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index]) #获取ee_tf

        # ik
        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])
        
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.ee_tf)], dtype=wp.vec3),#初始化目标位置为当前ee位置
        )

        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.ee_tf))], dtype=wp.vec4),#初始化目标姿态为当前ee姿态
        )

        self.obj_joint_limit = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        self.joint_q = self.model.joint_q.reshape((1, self.model.joint_coord_count))

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limit],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

# endregion

        # Modified: Added ZMQ subscriber setup
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect("tcp://localhost:5555")
        self.zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")

        self.capture()


    def capture(self):
        self.capture_ik()
        self.capture_sim()

    def capture_sim(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def capture_ik(self):
        self.graph_ik = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)
            self.graph_ik = cap.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contact, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    # Modified: Added method to update targets from ZMQ subscriber
    def _update_targets_from_zmq(self):
        """Receive pose from ZMQ and update IK objectives."""
        try:
            msg = self.zmq_socket.recv(zmq.NOBLOCK)
            pose_str = msg.decode('utf-8')
            parts = pose_str.split()
            if len(parts) == 7:
                pose = [float(p) for p in parts]
                pos = wp.vec3(pose[0], pose[1], pose[2])
                rot = wp.vec4(pose[3], pose[4], pose[5], pose[6])
                self.pos_obj.set_target_position(0, pos)
                self.rot_obj.set_target_rotation(0, rot)

                if self.graph_ik:
                    wp.capture_launch(self.graph_ik)
                else:                    
                    self.ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)
        except zmq.Again:
            pass  # No message available
        except ValueError:
            pass  # Invalid message format

    def step(self):
        # Modified: Changed to update targets from ZMQ instead of gizmos
        self._update_targets_from_zmq()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        # Visualize the current articulated state.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        # body_q_np = self.state_0.body_q.numpy()
        # Register gizmo (viewer will draw & mutate transform in-place).
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
        # wp.synchronize() #强制CPU等待GPU执行完成所有任务

if __name__ == "__main__":
    viewer = newton.viewer.ViewerGL()
    example = IKExample(viewer)
    while viewer.is_running():
        if not viewer.is_paused():
            example.step()
        example.render()
        
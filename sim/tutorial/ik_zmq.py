import warp as wp
import newton
import newton.ik as ik
import os
import zmq

# @wp.kernel
# def set_target_pose_kernel(pos_obj: ik.IKObjectivePosition, target_pos: wp.vec3):
#     pass
@wp.kernel
def set_joint_control_kernel(
    ik_solution: wp.array2d[wp.float32],
    joint_target: wp.array[wp.float32],
):
    # joint_target[:8] = ik_solution[0, :8] 在warp中切片赋值非法
    for i in range(8): # 只能逐个赋值
        joint_target[i] = ik_solution[0, i]

class IKExample:
    def __init__(self, viewer):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.graph = None
        # self.world_num = 2
        # self.robot_num = 1
        # zmq
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect("tcp://localhost:5555")
        self.zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")

        # build scene and robot
        robot = self.build_robot()
        scene = self.build_scene(robot)

        self.model_single = robot.finalize()
        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

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

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0) #更新state.body_q

        self.contacts = self.model.contacts()
        self.state = self.model.state()
        self.setup_ik()
        self.viewer.set_model(self.model) # viewer可视化 不做这步看不到模型
        self.capture()

    def capture(self):
        self.capture_sim()
        self.capture_ik()

    def build_robot(self):
        builder = newton.ModelBuilder()
        # newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        urdf_path = os.path.join(os.path.dirname(__file__), "assets/robot/ARX-X5/X5A.urdf")
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
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
        return builder

    def build_scene(self, robot:newton.ModelBuilder):
        builder = newton.ModelBuilder()
        builder.add_builder(robot)
        builder.add_ground_plane()
        return builder

    def setup_ik(self):
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

        self.joint_q_ik = self.model.joint_q.reshape((1, self.model_single.joint_coord_count))

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limit],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def capture_sim(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def capture_ik(self):
        self.graph_ik = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
            self.graph_ik = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def step(self):
        self.set_joint_target()
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
        except zmq.Again:
            pass  # No message available
        except ValueError:
            pass  # Invalid message format
        return self.pos_obj, self.rot_obj

    def set_joint_target(self):  
        # set target pose
        self.pos_obj, self.rot_obj = self._update_targets_from_zmq()
        # step ik solver
        if self.graph_ik:
            wp.capture_launch(self.graph_ik)
        else:            
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        # set joint target position
        wp.launch(
            set_joint_control_kernel,
            dim=1,
            inputs=[
                self.joint_q_ik,
                self.control.joint_target_pos,
            ],
        )


# multiworld就会需要两个model 一个是sim的model，一个是ik的model
if __name__ == "__main__":
    viewer = newton.viewer.ViewerGL()
    example = IKExample(viewer)
    while viewer.is_running():
        if not viewer.is_paused():
            example.step()
        example.render()
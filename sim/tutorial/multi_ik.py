import warp as wp
import numpy as np
import newton
import newton.ik as ik
import os
import zmq

@wp.kernel
def broadcast_ik_solution_kernel(
    ik_solution: wp.array2d[wp.float32],
    gripper_value: float,
    num_arm_joints: int,
    num_gripper_joints: int,
    robot_id: int,
    joint_targets: wp.array[wp.float32],
):
    # world_idx = wp.tid()
    num_total_joints = num_arm_joints + num_gripper_joints
    for j in range(num_arm_joints):
        joint_targets[robot_id * num_total_joints + j] = ik_solution[0, j]
    for j in range(num_gripper_joints):
        joint_targets[robot_id * num_total_joints + num_arm_joints + j] = gripper_value


class MutiIKExample:
    def __init__(self, viewer):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.graph = None

        #zmq
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect("tcp://localhost:5555")
        self.zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")
        #muti set
        self.num_robot = 2
        self.robot_1_pose = wp.transform(wp.vec3(0.0,0.5,0.0), wp.quat_identity())
        self.robot_2_pose = wp.transform(wp.vec3(0.0,-0.5,0.0), wp.quat_identity())
        self.robot_pose = [self.robot_1_pose, self.robot_2_pose]
        self.target = np.array(
            [
                0.50, 0.1, 0.2, 0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4), 0.04,
                0.50, -0.1, 0.2, 0.0, 0.0, -np.sin(np.pi / 4), np.cos(np.pi / 4), 0.04,
            ], dtype=np.float32
        )  # ee:xyzw + gripper

        #build scene
        self._single_robot_models = []
        scene = self.build_scene()
        self.build_robot(scene)

        # self.single_robot_model = robot.finalize()
        self.model = scene.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.state = self.model.state()

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

        # ready
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0) #更新state.body_q
        self.setup_ik()
        self.viewer.set_model(self.model) # viewer可视化 不做这步看不到模型
        self.capture()

    def build_robot(self, scene:newton.ModelBuilder):
        for robot_index in self.robot_pose: 
            builder = newton.ModelBuilder()
            # newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            urdf_path = os.path.join(os.path.dirname(__file__), "assets/robot/ARX-X5/X5A.urdf")
            builder.add_urdf(
                urdf_path,
                xform=robot_index,
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

            self._single_robot_models.append(builder.finalize())
            scene.add_builder(builder)

    def build_scene(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        return builder

    def capture(self):
        self.capture_sim()
        self.capture_ik()

    def capture_sim(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def capture_ik(self):
        self.graph_ik_list = []
        for robot_index in range(self.num_robot):
            with wp.ScopedCapture() as capture:
                self.ik_solvers[robot_index].step(
                    self.joint_q_ik_list[robot_index], self.joint_q_ik_list[robot_index],
                    iterations=self.ik_iters,
                )
            self.graph_ik_list.append(capture.graph)

    def step(self):
        self._update_targets_from_zmq()
        self.ik_compute_joint_target(self.target)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def simulate(self):
        self.state_0.clear_forces()
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def setup_ik(self):
        self.ik_iters = 24
        self.pos_objs = []
        self.rot_objs = []
        self.obj_joint_limits = []
        self.joint_q_ik_list = []
        self.ik_solvers = []
        self.ee_index = 6
        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])
        
        for model in self._single_robot_models:
            state = model.state()
            body_q_np = state.body_q.numpy()
            ee_tf = wp.transform(*body_q_np[self.ee_index])

            pos_obj = ik.IKObjectivePosition(
                link_index=self.ee_index,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.array([wp.transform_get_translation(ee_tf)], dtype=wp.vec3),
            )
            rot_obj = ik.IKObjectiveRotation(
                link_index=self.ee_index,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.array([_q2v4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4),
            )
            obj_joint_limits = ik.IKObjectiveJointLimit(
                joint_limit_lower=model.joint_limit_lower,
                joint_limit_upper=model.joint_limit_upper,
            )
            joint_q_ik = wp.array(model.joint_q, shape=(1, model.joint_coord_count))
            ik_solver = ik.IKSolver(
                model=model,
                n_problems=1,
                objectives=[pos_obj, rot_obj, obj_joint_limits],
                lambda_initial=0.1,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
            )

            self.pos_objs.append(pos_obj)
            self.rot_objs.append(rot_obj)
            self.obj_joint_limits.append(obj_joint_limits)
            self.joint_q_ik_list.append(joint_q_ik)
            self.ik_solvers.append(ik_solver)

    def ik_compute_joint_target(self, target: np.ndarray) -> None:
        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])
        for robot_index in range(self.num_robot):
            target_position = target[robot_index * 8 : robot_index * 8 + 3] # one robot's joint_num = 8
            target_rotation = target[robot_index * 8 + 3 : robot_index * 8 + 7]  # xyzw
            gripper_target = target[robot_index * 8 + 7]

            self.pos_objs[robot_index].set_target_positions(wp.array([target_position], dtype=wp.vec3))
            self.rot_objs[robot_index].set_target_rotations(wp.array([_q2v4(target_rotation)], dtype=wp.vec4))

            if self.graph_ik_list:
                wp.capture_launch(self.graph_ik_list[robot_index])
            else:
                self.ik_solvers[robot_index].step(
                    self.joint_q_ik_list[robot_index], self.joint_q_ik_list[robot_index],
                    iterations=self.ik_iters,
                )

            wp.launch(
                broadcast_ik_solution_kernel,
                dim=1,
                inputs=[
                    self.joint_q_ik_list[robot_index],
                    gripper_target,
                    6,# num_arm_joints
                    2,# num_gripper_joints
                    robot_index,
                    self.control.joint_target_pos  # output
                ],
            )

    def _update_targets_from_zmq(self):
        """Receive pose from ZMQ and update IK objectives."""
        try:
            # 非阻塞接收
            msg = self.zmq_socket.recv(zmq.NOBLOCK)
            # 转成字符串
            pose_str = msg.decode('utf-8')
            # 分割字符串成单个数字
            parts = pose_str.strip().split()
            if len(parts) != 16:
                print(f"Warning: Expected 16 numbers, got {len(parts)}")
                return
            # 转换成 float
            numbers = [float(x) for x in parts]
            # 转成 numpy array（float32）
            self.target = np.array(numbers, dtype=np.float32)
            # # 可选：分开两臂，方便调试
            # self.target_arm1 = self.target[:8]
            # self.target_arm2 = self.target[8:]
        except zmq.Again:
            pass
        except Exception as e:
            print(f"Error while receiving ZMQ message: {e}")



if __name__ == "__main__":
    viewer = newton.viewer.ViewerGL()
    example = MutiIKExample(viewer)
    while viewer.is_running():
        if not viewer.is_paused():
            example.step()
        example.render()
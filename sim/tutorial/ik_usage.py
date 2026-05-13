import warp as wp
import newton
import newton.ik as ik
import os

# 这份代码没有使用mujoco solver而是只使用了ik solver
# 直接写入joint_q，而没有写入control
# 所以viewer中看不到移动过程

class IKExample:
    def __init__(self, viewer):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
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
        self.viewer.set_model(self.model)
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state) #更新state.body_q

        # ee
        self.ee_index = 6
        body_q_np = self.state.body_q.numpy()
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
        self.solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limit],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def simulate(self):
        self.solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)

    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transform and push into IK objectives."""
        pos = wp.transform_get_translation(self.ee_tf)
        self.pos_obj.set_target_position(0, pos)
        q = wp.transform_get_rotation(self.ee_tf)
        self.rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    def step(self):
        self._push_targets_from_gizmos()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        # Visualize the current articulated state.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        body_q_np = self.state.body_q.numpy()
        # Register gizmo (viewer will draw & mutate transform in-place).
        self.viewer.log_gizmo("target_tcp", self.ee_tf, snap_to=wp.transform(*body_q_np[self.ee_index]))
        self.viewer.log_state(self.state)
        self.viewer.end_frame()
        wp.synchronize() #强制CPU等待GPU执行完成所有任务

if __name__ == "__main__":
    viewer = newton.viewer.ViewerGL()
    example = IKExample(viewer)
    while viewer.is_running():
        if not viewer.is_paused():
            example.step()
        example.render()
        
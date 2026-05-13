"""IK 控制器 — 基于 Newton IK 求解器的双臂逆运动学控制。"""

import numpy as np
import warp as wp
import newton.ik as ik

from .base_controller import BaseController
from utils.env_utils import quat_to_vec4, scatter_ik_solutions_kernel


class IKController(BaseController):
    """双臂 IK 控制器。

    为每个机械臂创建独立的 IK 求解器，接收 (num_envs, 16) 的目标数组：
    [left_pos(3), left_quat_xyzw(4), left_gripper(1),
     right_pos(3), right_quat_xyzw(4), right_gripper(1)]
    """

    IK_ITERS: int = 24

    def initialize_resources(self, env) -> None:
        self.setup_ik(env)

        # 捕获 IK CUDA Graph
        self.graph_ik_list = []
        if env._use_graph and wp.get_device().is_cuda:
            for ri in range(env.num_robot):
                with wp.ScopedCapture() as capture:
                    self.ik_solvers[ri].step(
                        self.joint_q_ik_list[ri],
                        self.joint_q_ik_list[ri],
                        iterations=self.IK_ITERS,
                    )
                self.graph_ik_list.append(capture.graph)
        else:
            self.graph_ik_list = [None] * env.num_robot

    def setup_ik(self, env) -> None:
        """为 env._single_arm_models 中每个模型构建独立的 IKSolver。"""
        assert len(env._single_arm_models) == env.num_robot, (
            f"需要 {env.num_robot} 个单臂模型，实际 {len(env._single_arm_models)} 个。"
        )

        self.n_problems = 1
        self.dofs_per_world = env.model.joint_dof_count

        self.pos_objs = []
        self.rot_objs = []
        self.obj_joint_limits_list = []
        self.joint_q_ik_list = []
        self.ik_solvers = []
        self.gripper_values_list = []

        for model_single in env._single_arm_models:
            state_single = model_single.state()
            body_q_np = state_single.body_q.numpy()
            ee_tf = wp.transform(*body_q_np[env.endeffector_id])

            init_pos = wp.transform_get_translation(ee_tf)
            init_rot = quat_to_vec4(wp.transform_get_rotation(ee_tf))

            pos_obj = ik.IKObjectivePosition(
                link_index=env.endeffector_id,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.array([init_pos] * self.n_problems, dtype=wp.vec3),
            )
            rot_obj = ik.IKObjectiveRotation(
                link_index=env.endeffector_id,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.array([init_rot] * self.n_problems, dtype=wp.vec4),
            )
            obj_joint_limits = ik.IKObjectiveJointLimit(
                joint_limit_lower=model_single.joint_limit_lower,
                joint_limit_upper=model_single.joint_limit_upper,
            )

            init_q_np = model_single.joint_q.numpy().astype(np.float32)
            joint_q_ik = wp.array(
                np.tile(init_q_np, (self.n_problems, 1)),
                dtype=wp.float32,
            )

            ik_solver = ik.IKSolver(
                model=model_single,
                n_problems=self.n_problems,
                objectives=[pos_obj, rot_obj, obj_joint_limits],
                lambda_initial=0.1,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
            )

            gripper_values = wp.zeros(self.n_problems, dtype=wp.float32)

            self.pos_objs.append(pos_obj)
            self.rot_objs.append(rot_obj)
            self.obj_joint_limits_list.append(obj_joint_limits)
            self.joint_q_ik_list.append(joint_q_ik)
            self.ik_solvers.append(ik_solver)
            self.gripper_values_list.append(gripper_values)

    def compute(self, env, target: np.ndarray) -> None:
        """对给定目标位姿求解 IK，将结果写入 env.control.joint_target_pos。

        target: (num_envs, num_robot * 8) 或 (num_robot * 8,)（单环境时自动升维）
        """
        target = np.asarray(target, dtype=np.float32)
        if target.ndim == 1:
            target = target[np.newaxis, :]  # (1, num_robot * 8)

        for ri in range(env.num_robot):
            target_positions = target[:, ri * 8: ri * 8 + 3]        # (num_envs, 3)
            target_rotations = target[:, ri * 8 + 3: ri * 8 + 7]    # (num_envs, 4) xyzw
            gripper_targets = target[:, ri * 8 + 7]                 # (num_envs,)

            self.pos_objs[ri].set_target_positions(
                wp.array([wp.vec3(*p) for p in target_positions], dtype=wp.vec3)
            )
            self.rot_objs[ri].set_target_rotations(
                wp.array([quat_to_vec4(q) for q in target_rotations], dtype=wp.vec4)
            )
            self.gripper_values_list[ri].assign(gripper_targets)

            if self.graph_ik_list[ri] is not None:
                wp.capture_launch(self.graph_ik_list[ri])
            else:
                self.ik_solvers[ri].step(
                    self.joint_q_ik_list[ri],
                    self.joint_q_ik_list[ri],
                    iterations=self.IK_ITERS,
                )

            wp.launch(
                scatter_ik_solutions_kernel,
                dim=self.n_problems,
                inputs=[
                    self.joint_q_ik_list[ri],
                    self.gripper_values_list[ri],
                    env.num_arm_joints,
                    env.num_gripper_joints,
                    ri,
                    self.dofs_per_world,
                    env.control.joint_target_pos,
                ],
            )

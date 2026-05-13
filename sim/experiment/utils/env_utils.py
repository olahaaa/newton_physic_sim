"""工具函数 — Warp kernel、资源路径解析。"""

from pathlib import Path
import warp as wp


# ═══════════════════════════════════════════════
# 资源路径
# ═══════════════════════════════════════════════

def resolve_asset_path(rel_path: str) -> Path:
    """将相对路径（相对于 sim/experiment/）转为绝对路径。"""
    base = Path(__file__).parents[1]  # sim/experiment/
    return base / rel_path


# ═══════════════════════════════════════════════
# 四元数工具
# ═══════════════════════════════════════════════

def quat_to_vec4(q: wp.quat) -> wp.vec4:
    """Warp quaternion → vec4 (xyzw)。"""
    return wp.vec4(q[0], q[1], q[2], q[3])


# ═══════════════════════════════════════════════
# Warp kernels — IK 解算分发
# ═══════════════════════════════════════════════

@wp.kernel
def scatter_ik_solutions_kernel(
    ik_solution: wp.array2d(dtype=wp.float32),
    gripper_values: wp.array(dtype=wp.float32),
    num_arm_joints: int,
    num_gripper_joints: int,
    robot_id: int,
    dofs_per_world: int,
    joint_targets: wp.array(dtype=wp.float32),
):
    world_idx = wp.tid()
    num_total_joints = num_arm_joints + num_gripper_joints
    base = world_idx * dofs_per_world + robot_id * num_total_joints
    for j in range(num_arm_joints):
        joint_targets[base + j] = ik_solution[world_idx, j]
    for j in range(num_gripper_joints):
        joint_targets[base + num_arm_joints + j] = gripper_values[world_idx]

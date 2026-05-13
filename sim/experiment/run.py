"""入口脚本 — 启动双臂衣物操控仿真，IK 控制。"""

import numpy as np
import warp as wp

from env.cloth_env_arx import ClothEnvARX


def main() -> None:
    # 创建环境（全部资产在构造时硬编码加载）
    env = ClothEnvARX(headless=False)

    # 两阶段初始化：创建 IK controller
    env.initialize_resources()

    # 初始 IK 目标（双臂各 8 维：pos(3) + quat_xyzw(4) + gripper(1)）
    # 左臂
    left_target = np.array([
        0.50, 0.10, 0.20,    # position
        0.0, 0.0, 0.0, 1,  # quat xyzw: z轴45°
        0.04,                 # gripper open
    ], dtype=np.float32)

    # 右臂
    right_target = np.array([
        0.50, -0.10, 0.20,   # position
        0.0, 0.0, 0, 1,  # quat xyzw: z轴-45°
        0.04,                 # gripper open
    ], dtype=np.float32)

    target = np.concatenate([left_target, right_target])  # (16,)

    # 主循环
    while env.viewer.is_running():
        if not env.viewer.is_paused():
            env.step(target)
        env.render()


if __name__ == "__main__":
    main()

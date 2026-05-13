"""入口脚本 — 启动双臂衣物操控仿真，ZMQ 通信控制。"""

import zmq
import numpy as np

from env.cloth_env_arx import ClothEnvARX


def main() -> None:
    # ZMQ SUB socket
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt(zmq.SUBSCRIBE, b"")

    # 创建环境（全部资产在构造时硬编码加载）
    env = ClothEnvARX(headless=False)

    # 两阶段初始化：创建 IK controller
    env.initialize_resources()

    # 默认初始目标（双臂各 8 维：pos(3) + quat_xyzw(4) + gripper(1)）
    target = np.array([
        0.50, 0.10, 0.20, 0.0, 0.0, 0.0, 1.0, 0.04,   # 左臂
        0.50, -0.10, 0.20, 0.0, 0.0, 0.0, 1.0, 0.04,  # 右臂
    ], dtype=np.float32)

    # 主循环
    while env.viewer.is_running():
        try:
            msg = socket.recv(zmq.NOBLOCK)
            parts = msg.decode().strip().split()
            if len(parts) != 16:
                print(f"Invalid input: need 16 floats, got {len(parts)}")
            else:
                target = np.array([float(p) for p in parts], dtype=np.float32)
        except zmq.Again:
            pass

        if not env.viewer.is_paused():
            env.step(target)
        env.render()


if __name__ == "__main__":
    main()

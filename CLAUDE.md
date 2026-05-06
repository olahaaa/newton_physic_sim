# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本仓库是基于 [Newton](https://github.com/olahaaa/newton.git)（GPU 加速物理引擎）的仿真环境，用于机器人仿真。Newton 以 git submodule 形式引入（`newton/` 目录），仿真脚本放在 `sim/` 目录下。

## 常用命令

所有命令使用 `uv`，在仓库根目录执行：

```bash
# 安装依赖（在 newton/ 子目录）
cd newton && uv sync --extra examples

# 运行示例
uv run -m newton.examples basic_pendulum

# 运行本仓库的仿真脚本
python sim/tutorial/0.basic_usage.py

# 运行测试
uv run --extra dev -m newton.tests                          # 全部测试
uv run --extra dev -m newton.tests -k <test_name>           # 单个测试

# 代码检查/格式化
uvx pre-commit run -a
```

Python 虚拟环境位于 `.venv/`，已安装 newton 及相关依赖（warp-lang、pyglet、mujoco 等）。

## 架构概览

### Newton 物理引擎核心概念

Newton 的仿真流程遵循固定模式：

1. **ModelBuilder** — 构建场景，添加物体（刚体、URDF 机器人、布料 mesh 等）
2. **Model** — `builder.finalize()` 生成模型，包含所有物体、关节、形状信息
3. **Solver** — 物理求解器（`SolverXPBD`、`SolverMuJoCo`、`SolverStyle3D`），每步推进仿真
4. **State / Control / Contacts** — 双缓冲状态模式：`state_0` 和 `state_1` 每步交换，避免 GPU 同步开销
5. **Viewer** — `newton.viewer.ViewerGL` 提供交互式渲染窗口

### 仿真循环模板

```python
state_0, state_1 = model.state(), model.state()
control, contacts = model.control(), model.contacts()

# CUDA 图捕获（可选，加速 GPU 执行）
with wp.ScopedCapture() as capture:
    simulate()
graph = capture.graph

while viewer.is_running():
    if not viewer.is_paused():
        wp.capture_launch(graph)  # 或 simulate()
    viewer.begin_frame(sim_time)
    viewer.log_state(state_0)
    viewer.end_frame()
```

### `sim/tutorial/` 脚本层次

| 文件 | 功能 | 使用的求解器 |
|------|------|-------------|
| `0.basic_usage.py` | 基础刚体掉落（球、胶囊、圆柱、mesh） | SolverXPBD |
| `1.cloth_usage.py` | 布料仿真 | SolverStyle3D |
| `2.urdf_usage.py` | URDF 机器人导入 + 碰撞管线 + 关节控制 | SolverMuJoCo |
| `ik_usage.py` | 单臂逆运动学（IK），支持 gizmo 拖动 | IKSolver |
| `multi_ik.py` | 双臂 IK + ZMQ 外部位姿控制 | SolverMuJoCo + IKSolver |
| `zmq_server.py` | ZMQ PUB 端，发送双臂目标位姿 | 无 |

### 路径使用规则

**必须**使用基于脚本目录的绝对路径加载资源文件，Python 的 CWD 不保证是脚本所在目录：

```python
import os
urdf_path = os.path.join(os.path.dirname(__file__), "assets/robot/ARX-X5/X5A.urdf")
```

### Newton 模块结构（`newton/newton/`）

- `geometry.py` — 几何类型（`GeoType`、`HydroelasticSDF`）
- `solvers.py` — 物理求解器（`SolverXPBD`、`SolverMuJoCo`、`SolverStyle3D`）
- `ik.py` — 逆运动学（`IKSolver`、IK 目标约束）
- `viewer.py` — OpenGL 交互式查看器
- `usd.py` — USD 文件导入导出
- `math.py` — 数学工具
- `_src/` — **内部实现，教程和示例代码不得直接导入**

### Newton 开发规范（摘自 `newton/AGENTS.md`）

- 公开 API 符号通过顶层模块暴露，不从 `newton._src` 导入
- 破坏性 API 变更需先标记 deprecated
- 前缀优先命名：`ActuatorPD` 而非 `PDActuator`
- 类型标注用 PEP 604 联合语法（`x | None`）
- Warp 数组用括号语法（`wp.array[wp.vec3]`）
- Google 风格 docstrings，物理量标注 SI 单位
- 新增公开 API 符号后需运行 `docs/generate_api.py`
- 测试用 `unittest`，不用 pytest
- 提交信息用祈使语气，~50 字符标题

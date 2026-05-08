# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本项目使用 Newton Physics Engine（GPU 加速物理仿真引擎）进行机器人仿真，主要工作是 ARX-X5 机械臂的逆运动学（IK）控制。Newton 引擎以 git submodule 形式引入（`newton/`），用户自定义仿真代码在 `sim/` 目录。

**重要规则：`newton/` 目录是只读的第三方源码。** 本项目仅使用其 `pyproject.toml` 构建依赖并参考其部分源码，未做任何代码修改。禁止修改 `newton/` 内的任何文件，所有开发工作只能在 `sim/` 及项目根目录进行。

## 环境与依赖管理

使用 `uv` 管理 Python 依赖，目标 Python 版本 3.12。

```bash
# 同步依赖（在项目根目录运行）
uv sync --extra examples          # 安装仿真+可视化依赖
uv sync --extra dev               # 安装开发依赖（含测试）
uv sync --extra dev --extra torch-cu12  # 含 PyTorch (CUDA 12)

# 初始化/更新子模块
git submodule update --init --recursive
```

虚拟环境位于 `.venv/`。

## 运行仿真脚本

用户自定义脚本都在 `sim/tutorial/` 目录下。由于脚本使用相对路径加载资源文件（如 URDF、USD），**必须从 `sim/tutorial/` 目录内运行**，否则会找不到资源文件：

```bash
cd sim/tutorial
uv run python multi_ik.py        # 多机器人 IK 控制
uv run python ik_usage_zmq.py    # 单机器人 IK + ZMQ 通信
uv run python ik_usage.py        # 单机器人 IK
uv run python 0.basic_usage.py   # 基础用法示例
uv run python 1.cloth_usage.py   # 布料仿真示例
uv run python 2.urdf_usage.py    # URDF 加载示例
```

## 运行 Newton 官方示例

```bash
uv run -m newton.examples basic_pendulum
uv run -m newton.examples basic_urdf
uv run -m newton.examples basic_shapes
```

## 测试

Newton 使用 `unittest`（非 pytest）：

```bash
uv run --extra dev -m newton.tests                                    # 全部测试
uv run --extra dev -m newton.tests -k test_viewer_log_shapes         # 特定测试
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes # 示例测试
```

## 代码检查

```bash
uvx pre-commit run -a   # ruff lint+format, typos, uv-lock 检查
```

## 架构

### 仓库结构

```
newton_physic_sim/
  newton/                  # Newton 物理引擎（git submodule，fork 自 olahaaa/newton）
    newton/_src/           # 内部实现（禁止从 sim/ 直接导入）
    newton/geometry.py     # 公共 API：碰撞检测相关
    newton/solvers.py      # 公共 API：8 种求解器
    newton/ik.py           # 公共 API：逆运动学
    newton/viewer.py       # 公共 API：可视化后端
    ...
  sim/tutorial/            # 用户仿真脚本
    assets/robot/ARX-X5/   # ARX-X5 机械臂 URDF 和网格文件
    assets/cloth/          # 布料仿真资源
    notes/                 # 开发笔记与报错记录
```

### 核心概念

- **Model** — 静态仿真定义（几何体、刚体、关节等），通过 `ModelBuilder` 构建
- **State** — 时变量（位置、速度、力），通过 `model.state()` 创建
- **Control** — 驱动信号（关节力、目标位置），通过 `model.control()` 创建
- **Contacts** — 运行时碰撞接触数据
- **Solver** — 8 种求解器可选，本项目中主要使用 `SolverMuJoCo`（integrator="implicitfast"）
- **IK** — `newton.ik.IKSolver` 提供 GPU 加速的逆运动学求解，支持位置/旋转/关节限制目标

### 仿真流程

1. `ModelBuilder` 构建场景（地面、URDF 加载、多机器人组合）
2. 选择一个 Solver
3. 调用 `solver.step(state_0, state_1, control, contacts, dt)` 推进仿真
4. 使用 `ViewerGL` 进行可视化

### Newton 公共 API 规范（来自 AGENTS.md）

- 禁止从 `newton._src` 导入，所有公共符号通过顶层模块暴露
- 前缀优先命名：`ActuatorPD`、`add_shape_sphere()`
- 使用 PEP 604 联合类型（`x | None`）
- Warp 数组注解使用方括号语法：`wp.array[wp.vec3]`
- Google 风格文档字符串，物理量使用 SI 单位
- 破坏性变更需要先声明弃用

## ZMQ 通信

`multi_ik.py` 和 `ik_usage_zmq.py` 使用 ZMQ SUB socket 接收外部控制指令，默认连接 `tcp://localhost:5555`。预期接收 16 个空格分隔的浮点数（双臂各 8 个值：xyz + quaternion xyzw + gripper_value）。可运行 `zmq_server.py` 作为测试用的 ZMQ 发送端。

## 硬件环境

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU (8 GiB)
- CUDA Toolkit 12.9, Driver 13.0
- Warp kernel 缓存: `~/.cache/warp/1.12.1`

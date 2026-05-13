# request2 解答

## `cloth_env_arx.py` 第 133-152 行分析

### 第 133-146 行：重力补偿块

```python
if self.GRAVITY_COMPENSATION:
    newton.solvers.SolverMuJoCo.register_custom_attributes(robot_builder)
    gravcomp_jnt = robot_builder.custom_attributes["mujoco:jnt_actgravcomp"]
    ...
    gravcomp_body = robot_builder.custom_attributes["mujoco:gravcomp"]
    ...
```

**作用**：通过设置 MuJoCo 的 `jnt_actgravcomp` 和 `gravcomp` 自定义属性，对机械臂关节进行重力补偿。在 armature 较高的关节上可抵消重力产生的偏转。

**实际使用情况**：`GRAVITY_COMPENSATION` 类变量固定为 `False`（第 41 行），整个块永远不会执行。当前仿真中未启用重力补偿。

**结论**：可以删除。如果将来需要重力补偿，再按需加回即可。

---

### 第 147-152 行：`b2w_list`（基座到世界位姿记录）

```python
pos_np = np.array([xform.p[0], xform.p[1], xform.p[2]], dtype=np.float32)
quat_np = np.array([xform.q[0], xform.q[1], xform.q[2], xform.q[3]], dtype=np.float32)
base_pose = np.concatenate([pos_np, quat_np])
self.b2w_list = getattr(self, "b2w_list", [])
self.b2w_list.append(xyzw_to_wxyz(base_pose))
```

**作用**：将每个机械臂的 URDF 基座位姿（xyzw 四元数格式）转换为 wxyz 格式，存入 `self.b2w_list`。在主项目 `termitech_cloth_sim` 中，这个列表被 RL 环境用于将 IK 解算结果从局部关节空间映射回全局坐标。

**实际使用情况**：
- `b2w_list` **只在当前文件第 151-152 行写入，从未被读取**。
- 全局 grep 确认：项目中没有任何其他代码引用 `b2w_list`。
- 这是从主项目 `termitech_cloth_sim` 移植时带过来的冗余代码。

**结论**：可以删除。

---

## 建议操作

删除 `cloth_env_arx.py` 中以下内容：

| 行号 | 内容 | 原因 |
|---|---|---|
| 41 | `GRAVITY_COMPENSATION: bool = False` | 始终为 False，从未启用 |
| 133-146 | 整个 `if self.GRAVITY_COMPENSATION:` 块 | 永不执行 |
| 148-152 | `base_pose` 构建 + `b2w_list` 追加 | 只写不读，无调用者 |
| 9 | `from utils.env_utils import xyzw_to_wxyz` | 删除 b2w_list 后不再需要 |

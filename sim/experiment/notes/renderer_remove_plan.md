# 移除 renderer 模块实施计划

## 目标

删除 `sim/experiment/renderer/` 模块（`base_renderer.py`、`default_renderer.py`、`__init__.py`），
将 `DefaultRenderer.render()` 的渲染逻辑直接并入 `BaseEnv.render()`。

## 影响分析

renderer 模块的外部引用仅限于 `base_env.py`：

| 引用 | 位置 | 操作 |
|---|---|---|
| `from renderer.base_renderer import BaseRenderer` | base_env.py:10 | 删除 |
| `from renderer.default_renderer import DefaultRenderer` | base_env.py:11 | 删除 |
| `self.renderer: Optional[BaseRenderer] = None` | base_env.py:31 | 删除 |
| `no_renderer` 参数 | base_env.py:37 | 删除 |
| `if not no_renderer: self.setup_renderer()` | base_env.py:39-40 | 删除 |
| `setup_renderer()` 方法 | base_env.py:48-51 | 删除 |
| `render()` 方法体（委托） | base_env.py:89-93 | 替换为直接渲染逻辑 |

`run.py` 和 `cloth_env_base.py` 不依赖 renderer 模块（`cloth_env_base.py:91` 的 `hasattr(self.viewer, "renderer")` 是 Newton 内部 viewer 属性）。

## 具体步骤

### 1. 修改 `base_env.py`

**删除导入**（第 10-11 行）：
- `from renderer.base_renderer import BaseRenderer`
- `from renderer.default_renderer import DefaultRenderer`

**删除属性**（第 31 行）：
- `self.renderer: Optional[BaseRenderer] = None`

**简化 `initialize_resources()`**：
- 移除 `no_renderer` 参数
- 移除 `setup_renderer()` 调用
- 只保留 `setup_controller()` 调用

**删除 `setup_renderer()` 方法**（第 48-51 行）。

**重写 `render()` 方法**（第 89-93 行）：
```python
def render(self, return_renderings: bool = False) -> dict:
    if self.viewer is None:
        return {}
    self.viewer.begin_frame(self.sim_time)
    self.viewer.log_state(self.state_0)
    self.viewer.end_frame()
    if return_renderings:
        rgb = self.viewer.get_frame().numpy()
        h, w = rgb.shape[:2]
        rgba = np.concatenate([rgb, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
        return {"rgba": rgba}
    return {}
```

### 2. 修改 `run.py`

更新注释：移除 "renderer" 提及。

### 3. 删除文件

```
rm sim/experiment/renderer/base_renderer.py
rm sim/experiment/renderer/default_renderer.py
rm sim/experiment/renderer/__init__.py
rm -rf sim/experiment/renderer/__pycache__
```

### 4. 不修改的文件

- `cloth_env_base.py`：`hasattr(self.viewer, "renderer")` 是 Newton ViewerGL 的内部属性，与 renderer 模块无关。
- `cloth_env_arx.py`：无引用。
- `controller/`：无引用。
- `utils/`：无引用。

## 预计效果

- 删除 3 个文件（renderer 模块）
- `base_env.py` 减少约 10 行间接调用代码
- `run.py` → `env.render()` 调用不变，渲染逻辑直接在 BaseEnv 中执行

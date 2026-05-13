# request3 解答

## `env.render()` 为何不传参数？

这是**两个不同类**的 `render` 方法：

| 调用 | 所属类 | 签名 | `self` 是谁 |
|---|---|---|---|
| `env.render()` | `BaseEnv` | `render(self, return_renderings=False)` | env 本身 |
| `renderer.render(self, env, ...)` | `DefaultRenderer` | `render(self, env, return_renderings=False)` | renderer 对象 |

调用链：

```
run.py: env.render()
  → BaseEnv.render(self)                         # self = env
    → self.renderer.render(self, ...)            # 把 env 作为 env 参数传入渲染器
      → DefaultRenderer.render(self, env, ...)   # self = renderer, env = 环境
```

`DefaultRenderer.render` 需要 `env` 参数是因为渲染器是独立对象，必须通过 `env.viewer` / `env.state_0` / `env.sim_time` 才能拿到要渲染的数据。`BaseEnv.render` 调用时把 `self`（即环境自身）传进去。

"""默认 GL 渲染器 — 直接使用 ViewerGL 可视化。"""

import numpy as np

from .base_renderer import BaseRenderer


class DefaultRenderer(BaseRenderer):

    def initialize_resources(self, env) -> None:
        pass

    def render(self, env, return_renderings: bool = False) -> dict:
        if env.viewer is None:
            return {}

        env.viewer.begin_frame(env.sim_time)
        env.viewer.log_state(env.state_0)
        env.viewer.end_frame()

        if return_renderings:
            rgb = env.viewer.get_frame().numpy()  # (H, W, 3), top-left origin
            h, w = rgb.shape[:2]
            rgba = np.concatenate([rgb, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
            return {"rgba": rgba}
        return {}

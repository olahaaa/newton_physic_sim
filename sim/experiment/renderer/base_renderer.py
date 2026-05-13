"""渲染器基类。"""


class BaseRenderer:

    def initialize_resources(self, env) -> None:
        pass

    def render(self, env, return_renderings: bool = False) -> dict:
        pass

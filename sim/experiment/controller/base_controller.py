"""控制器基类。"""

import numpy as np


class BaseController:

    def initialize_resources(self, env) -> None:
        pass

    def compute(self, env, target: np.ndarray) -> None:
        pass

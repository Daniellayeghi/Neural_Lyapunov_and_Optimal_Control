import torch
import numpy as np
from utilities.mj_renderer import MjRenderer


class PolicyVisualizer:
    def __init__(self, model, env, xml_path: str, transform_func=lambda x: x):
        self._model = model
        self._env = env
        self._renderer = MjRenderer(xml_path)
        self._transform_func = transform_func
        self._obs = None

    def _simulate(self):
        action, _ = self._model.predict(self._obs)
        res = self._env.step(np.clip(action, -20, 20))
        self._obs = res[0]
        pos_size = self._obs.flatten().shape[0] // 2
        return self._obs.flatten()[:pos_size]

    def visualize(self, horizon: int = 1000) -> None:
        try:
            while True:  # Keep running until interrupted
                self._obs = self._env.reset()
                positions = [self._simulate() for _ in range(horizon)]
                self._renderer.render(np.array(positions))

        except KeyboardInterrupt:  # Handle interruption
            print("Visualization stopped.")
            return


class ValueFunction:

    def __init__(self, model):
        self._model = model

    def __call__(self, state):
        state_tensor = torch.as_tensor(state, device=self._model.policy.device).unsqueeze(0)
        _, values, _ = self._model.policy.forward(state_tensor, deterministic=True)
        state_value = values.item()

        return state_value

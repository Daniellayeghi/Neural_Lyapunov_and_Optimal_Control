import torch
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from utilities.mj_renderer import MjRenderer


class PolicyVisualizer:
    def __init__(self, model, env, xml_path: str, transform_func=lambda x: x, limit=np.inf):
        self._model = model
        self._env = env
        self._renderer = MjRenderer(xml_path)
        self._transform_func = transform_func
        self._obs = None
        self._limit = limit

    def _simulate(self):
        action, _ = self._model.predict(self._obs)
        res = self._env.step(action)
        reset = res[-2][0]
        self._obs = res[0]
        pos_size = self._obs.flatten().shape[0] // 2
        return self._obs.flatten()[:pos_size], reset

    def visualize(self, horizon: int = 600) -> None:
        try:
            iteration = 0
            while iteration < self._limit:  # Keep running until interrupted
                self._obs = self._env.reset()
                positions = [pos for pos, reset in [self._simulate() for _ in range(horizon)] if not reset]
                results = np.array(positions)
                if self._transform_func is not None:
                    results = self._transform_func(results)

                self._renderer.render(results)
                iteration += 1

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


class VisualizePolicyCallback(EvalCallback):
    def __init__(self, eval_env, xml_path, visualize_nth, **kwargs):
        super().__init__(eval_env, **kwargs)
        self._visualizer = None

        self._vnth = visualize_nth
        if xml_path is None:
            raise ValueError("xml_path is required for VisualEvalCallback.")
        self._xml_path = xml_path

    @property
    def visualizer(self):
        if self._visualizer is None:
            self._visualizer = PolicyVisualizer(
                self.model, self.eval_env, self._xml_path, transform_func=self.eval_env.envs[0].transform_func, limit=1
            )
        return self._visualizer

    def _on_step(self) -> bool:
        res = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % (self.eval_freq * self._vnth) == 0:
            self.visualizer.visualize()
        return res

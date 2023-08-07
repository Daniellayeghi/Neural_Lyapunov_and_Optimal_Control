import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from utilities.mj_renderer import MjRenderer


def _make_env(model_type: gym.Env, seed, **model_params):
    def _f():
        env = model_type(**model_params)
        env.seed(seed)
        return env
    return _f


def make_subproc_vec_env(model_type: gym.Env, nproc, **model_params):
    envs = [_make_env(model_type=model_type, seed=seed, **model_params) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)
    return envs


class PolicyVisualizer:
    def __init__(self, model, env, xml_path: str):

        self.model = model
        self.env = env
        self.renderer = MjRenderer(xml_path)

    def visualize(self, horizon: int = 1000) -> None:
        obs, _ = self.env.reset()
        positions = []

        for _ in range(horizon):
            action, _ = self.model.predict(obs)
            obs, _, _, _, _ = self.env.step(action)
            # Assuming the first element of obs is the position
            positions.append(obs[0])

        # Render the sequence of positions
        self.renderer.render(np.array(positions))


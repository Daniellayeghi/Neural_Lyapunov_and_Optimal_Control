import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv


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

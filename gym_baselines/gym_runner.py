from multiprocessing import freeze_support
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from gym_baselines.gym_configs import configurations
from utilities.gym_utils import VisualizePolicyCallback


def _make_env(model_type: gym.Env, seed, **model_params):
    def _f():
        env = model_type(**model_params)
        env.seed(seed)
        return env

    return _f


def _make_subproc_vec_env(model_type: gym.Env, nproc, **model_params):
    envs = [_make_env(model_type=model_type, seed=seed, **model_params) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)
    return envs


def _setup_environment(env_name, nproc, model_params):
    envs = _make_subproc_vec_env(model_type=env_name, nproc=nproc, **model_params)
    eval_env = DummyVecEnv([lambda: env_name(**model_params)])
    return envs, eval_env


def _initialize_model(algorithm, envs, hyperparameters):
    if algorithm == "PPO":
        model = PPO(**hyperparameters, env=envs)
    elif algorithm == "SAC":
        model = SAC(**hyperparameters, env=envs)
    return model


def _run_learning_process(model, total_timesteps, eval_env, tb_name, eval_freq, xml_path):
    kwargs = {'log_path': tb_name, 'eval_freq': eval_freq, 'n_eval_episodes': 5}
    eval_callback = VisualizePolicyCallback(eval_env, xml_path, 10, **kwargs)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)


def _save_results(tb_name, env_name, algorithm):
    res = np.load(f"{tb_name}evaluations.npz")
    rewards = -res['results']
    path = f"./data/{env_name}_rewards_{algorithm}.csv"
    np.savetxt(path, rewards, delimiter=",")
    return path


def get_main_function(env, solver):
    def main_function():
        freeze_support()
        config = configurations[f"{env.upper()}_{solver.upper()}"]
        envs, eval_env = _setup_environment(config['model_type'], config['nproc'], config.get('model_params', {}))
        model = _initialize_model(solver.upper(), envs, config['hyperparameters'])
        total_timesteps = config['nproc'] * config['terminal_time'] * config['epochs']
        eval_freq = int(total_timesteps / (config['nproc'] * config['epochs']))

        # All the training loop happens here #
        _run_learning_process(
            model, total_timesteps, eval_env, config['hyperparameters']['tensorboard_log'], eval_freq,
            config['xml_path']
        )

        path = _save_results(config['hyperparameters']['tensorboard_log'], config['env_name'], solver)
        name = f"./models/{env.upper()}_{solver.upper()}_{config['epochs']}"
        model.policy.to("cpu")
        model.save(f"{name}")
        return path, name

    return main_function

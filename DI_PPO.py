from multiprocessing import freeze_support
from collections import OrderedDict
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_models import CustomDoubleIntegrator
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from utilities.gym_utils import make_subproc_vec_env
gym.logger.MIN_LEVEL = gym.logger.min_level

env_name = 'CustomDoubleIntegrator'
epochs, terminal_time, nproc = 80, 300, 6
total_timesteps = (nproc * terminal_time) * epochs
eval_freq = int(total_timesteps / (nproc * epochs))

print(f"Running {env_name} for {epochs} epochs resulting in {total_timesteps} total time.")


def main():
    freeze_support()

    model_params = {
        'env_id': env_name,
        'init_bound': (-3, 3),
        'terminal_time': 300,
    }

    envs = make_subproc_vec_env(model_type=CustomDoubleIntegrator, nproc=nproc, **model_params)
    eval_env = DummyVecEnv([lambda: CustomDoubleIntegrator(**model_params)])

    ppo_params = OrderedDict(
        [('ent_coef', 0.0),
         ('gae_lambda', 0.98),
         ('gamma', 0.99),
         ('n_epochs', 4),
         ('n_steps', 16),
         ('policy', 'MlpPolicy'),
         ('normalize_advantage', True)]
    )

    model = PPO(**ppo_params, env=envs, tensorboard_log="./ppo_tensorboard_di/", verbose=1)
    eval_callback = EvalCallback(eval_env, log_path="./ppo_tensorboard_di/", eval_freq=eval_freq, n_eval_episodes=5)
    model.learn(total_timesteps=int(total_timesteps), callback=eval_callback)
    envs.close()
    res = np.load("./ppo_tensorboard_di/evaluations.npz")
    rewards = res['results']
    path = f"./data/{env_name}_rewards_ppo.csv"
    np.savetxt(path, rewards, delimiter=",")

    return path


if __name__ == "__main__":
    main()
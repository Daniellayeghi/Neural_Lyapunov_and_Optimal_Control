from collections import OrderedDict
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_linear_fn
from gym_baselines.gym_models import CustomReacher
from stable_baselines3.common.callbacks import EvalCallback
from utilities.gym_utils import make_subproc_vec_env
gym.logger.MIN_LEVEL = gym.logger.min_level

env_name = 'Custom Reacher PPO'
tb_name = './ppo_tensorboard_tl/'
epochs, terminal_time, nproc = 80, 300, 6
total_timesteps = (nproc * terminal_time) * epochs
eval_freq = int(total_timesteps / (nproc * epochs))

print(f"Running {env_name} for {epochs} epochs resulting in {total_timesteps} total time.")


def main():
    freeze_support()

    model_params = {
        'env_id': env_name,
        'init_bound': {
            'position': (-3, 3),
            'velocity': (0, 0)
        },
        'terminal_time': 300,
    }

    envs = make_subproc_vec_env(model_type=CustomReacher, nproc=nproc, **model_params)
    eval_env = DummyVecEnv([lambda: CustomReacher(**model_params)])
    eval_env.seed(np.random.randint(20))

    initial_clip_range = 1.0
    final_clip_range = 0.0
    end_fraction = 0.4

    clip_range_schedule = get_linear_fn(initial_clip_range, final_clip_range, end_fraction)

    ppo_params =OrderedDict(
        [('batch_size', 64),
         ('clip_range', clip_range_schedule),
         ('ent_coef', 0.0),
         ('gae_lambda', 0.9),
         ('gamma', 0.99),
         ('learning_rate', 3e-05),
         ('max_grad_norm', 0.5),
         ('n_epochs', 20),
         ('n_steps', 512),
         ('policy', 'MlpPolicy'),
         ('policy_kwargs',
          dict(log_std_init=-2.7, ortho_init=False,activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256]))),
         ('sde_sample_freq', 4),
         ('use_sde', True),
         ('vf_coef', 0.5),
         ('normalize_advantage', True)]
    )

    model = PPO(**ppo_params, env=envs, tensorboard_log=tb_name, verbose=1)
    eval_callback = EvalCallback(eval_env, log_path=tb_name, eval_freq=eval_freq, n_eval_episodes=5)
    model.learn(total_timesteps=int(total_timesteps), callback=eval_callback)

    envs.close()
    res = np.load(f"{tb_name}/evaluations.npz")
    rewards = res['results']
    path = f"./data/{env_name}_rewards_ppo.csv"
    np.savetxt(path, rewards, delimiter=",")

    return path


if __name__ == "__main__":
    from utilities.plotting import plot_reward_graph_multi

    res_path = main()
    plot_reward_graph_multi(res_path, env_name)
    plt.show()

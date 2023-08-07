from multiprocessing import freeze_support
from collections import OrderedDict
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_models import CustomReacher
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from utilities.gym_utils import make_subproc_vec_env

gym.logger.MIN_LEVEL = gym.logger.min_level

env_name = 'CustomRe'
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
    eval_env.seed(10)
    sac_params = OrderedDict(
        [('batch_size', 256),
         ('buffer_size', 300000),
         ('ent_coef', 'auto'),
         ('gamma', 0.98),
         ('gradient_steps', 64),
         ('learning_rate', 0.00073),
         ('learning_starts', 10000),
         ('policy', 'MlpPolicy'),
         ('policy_kwargs', dict(log_std_init=-3, net_arch=[400, 300])),
         ('tau', 0.02),
         ('train_freq', 64),
         ('use_sde', False)]
    )

    model = SAC(**sac_params, env=envs, tensorboard_log="./sac_tensorboard_tl/", verbose=1)
    eval_callback = EvalCallback(eval_env, log_path="./sac_tensorboard_tl/", eval_freq=eval_freq, n_eval_episodes=5)
    # episode_avg_reward_callback = EpisodeAverageRewardCallback(num_envs=nproc)
    model.learn(total_timesteps=int(total_timesteps), callback=eval_callback)
    # Close the environments after training is complete
    envs.close()
    res = np.load("./sac_tensorboard_tl/evaluations.npz")
    rewards = res['results']
    path = f"./data/{env_name}_rewards_sac.csv"
    np.savetxt(path, rewards, delimiter=",")

    return path


if __name__ == "__main__":
    from utilities.plotting import plot_reward_graph_multi

    res_path = main()
    plot_reward_graph_multi(res_path, "Reacher SAC")

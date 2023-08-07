
import gymnasium as gym
gym.logger.MIN_LEVEL = gym.logger.min_level
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gym_models import CustomDoubleIntegrator
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

env_name = 'CustomDoubleIntegrator'
epochs, terminal_time, nproc = 80, 300, 6
total_timesteps = (nproc * terminal_time) * epochs
eval_freq = int(total_timesteps / (nproc * epochs))

print(f"Running {env_name} for {epochs} epochs resulting in {total_timesteps} total time.")

from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()


    def make_env(env_id, seed):
        def _f():
            env = CustomDoubleIntegrator(env_id, (-3, 3), terminal_time, seed)
            env.seed(seed)
            return env

        return _f


    envs = [make_env(env_name, seed) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)

    eval_env = DummyVecEnv([lambda: CustomDoubleIntegrator(100, (-3, 3), terminal_time, 100)])

    model = SAC("MlpPolicy",
                envs,
                batch_size=512,
                buffer_size=50000,
                ent_coef=0.1,
                gamma=0.9999,
                gradient_steps=32,
                learning_rate=0.0003,
                learning_starts=0,
                tau=0.01,
                train_freq=32,
                use_sde=True,
                policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]),
                tensorboard_log="./sac_tensorboard_di/",
                verbose=1
                )

    eval_callback = EvalCallback(eval_env, log_path="./sac_tensorboard_di/", eval_freq=eval_freq, n_eval_episodes=5)
    # episode_avg_reward_callback = EpisodeAverageRewardCallback(num_envs=nproc)
    model.learn(total_timesteps=int(total_timesteps), callback=eval_callback)
    # Close the environments after training is complete
    envs.close()
    res = np.load("./sac_tensorboard_di/evaluations.npz")
    rewards = res['results']
    np.savetxt(f"{env_name}_rewards_sac.csv", rewards, delimiter=",")
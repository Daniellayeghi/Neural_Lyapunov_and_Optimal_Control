import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gym_baselines.gym_runner import _initialize_model
from utilities.gym_utils import PolicyVisualizer
from gym_baselines.gym_configs import configurations


def generate_model_and_env(env, solver, model_path=None):
    config = configurations[f"{env.upper()}_{solver.upper()}"]
    xml_path = config['xml_path']
    horizon = config['terminal_time']
    envs = DummyVecEnv([lambda: config['model_type'](**config.get('model_params', {}))])
    model = _initialize_model(solver.upper(), envs, config['hyperparameters'], model_path)
    return model, envs, xml_path, horizon


def main():
    parser = argparse.ArgumentParser(description="Visualize a policy using a given model and environment type.")
    parser.add_argument('--model', type=str, help="Path to the saved model.", required=False)
    parser.add_argument('--env', choices=['di', 'tl', 'cp_balance', 'cp_swingup'], help='Environment(s) to run', required=False)
    parser.add_argument('--solver', choices=['ppo', 'sac'], default='sac', help='ppo and/or sac')
    args = parser.parse_args()
    model, env, xml_path, terminal_time = generate_model_and_env(args.env, args.solver, args.model)
    print()
    #put model on cpu
    import torch
    model.policy.to(torch.device('cpu'))
    visualizer = PolicyVisualizer(model, env, xml_path, transform_func=env.envs[0].transform_func)
    visualizer.visualize(horizon=terminal_time)


if __name__ == "__main__":
    main()

import argparse
from gym_baselines.gym_runner import _make_subproc_vec_env
from gym_baselines.gym_runner import _initialize_model
from utilities.gym_utils import PolicyVisualizer
from gym_baselines.gym_configs import configurations


def generate_model_and_env(env, solver):
    config = configurations[f"{env.upper()}_{solver.upper()}"]
    envs = _make_subproc_vec_env(config['model_type'], 1, **config.get('model_params', {}))
    model = _initialize_model(solver.upper(), envs, config['hyperparameters'])
    return model, envs


def main():
    parser = argparse.ArgumentParser(description="Visualize a policy using a given model and environment type.")
    parser.add_argument('--model_path', type=str, default='./models/CP_BALANCE_SAC_100.zip', help="Path to the saved model.")
    parser.add_argument('--xml_path', type=str, default="./xmls/cartpole.xml", help="Path to the XML file for rendering.")
    parser.add_argument('--horizon', type=int, default=300, help="Time horizon for visualization.")
    parser.add_argument('--env', choices=['di', 'tl', 'cp_balance'], default='cp_balance', help='Environment(s) to run')
    parser.add_argument('--solver', choices=['ppo', 'sac'], default='sac', help='ppo and/or sac')
    args = parser.parse_args()

    model, env = generate_model_and_env(args.env, args.solver)
    visualizer = PolicyVisualizer(model, env, args.xml_path)
    visualizer.visualize(horizon=args.horizon)


if __name__ == "__main__":
    main()

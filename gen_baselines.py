import argparse
import multiprocessing
import matplotlib.pyplot as plt

from DI_PPO import main as main_ppo_di
from DI_SAC import main as main_sac_di
from TL_SAC import main as main_sac_tl
# Add import for TL_PPO once you have it
# from TL_PPO import main as main_ppo_tl

from utilities.plotting import plot_reward_graph_multi


def wrapper(func, queue):
    queue.put(func())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run reinforcement learning algorithms on specified environments.')
    parser.add_argument(
        '--env', choices=['di', 'tl'], required=True, nargs='+', help='Environment(s) to run: "di" and/or "tl"'
    )
    parser.add_argument(
        '--solver', choices=['ppo', 'sac'], required=True, nargs='+', help='Solver(s) to use: "ppo" and/or "sac"'
    )
    args = parser.parse_args()

    functions = {
        'di': {
            'ppo': main_ppo_di,
            'sac': main_sac_di
        },
        'tl': {
            'ppo': None,  # main_ppo_tl once you have it
            'sac': main_sac_tl
        }
    }

    queues = [multiprocessing.Queue() for _ in args.env for _ in args.solver]
    procs = [multiprocessing.Process(target=wrapper, args=(functions[env][solver], q))
             for env in args.env for solver in args.solver for q in queues]

    [p.start() for p in procs]
    [p.join() for p in procs]

    results = [q.get() for q in queues]

    print("Generating plots")
    for result, env, solver in zip(results, args.env*len(args.solver), args.solver*len(args.env)):
        plot_reward_graph_multi(result, f"{env.upper()} {solver.upper()}")
    plt.show()

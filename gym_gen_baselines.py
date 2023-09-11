import argparse
import multiprocessing
import matplotlib.pyplot as plt

from utilities.plotting import plot_reward_graph_multi
from gym_baselines.gym_runner import get_main_function


def wrapper(func, queue):
    queue.put(func())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run reinforcement learning algorithms on specified environments.')
    parser.add_argument(
        '--env', choices=['di', 'tl', 'cp_balance', 'cp_swingup'], default=['tl'], nargs='+', help='Environment(s) to run'
    )
    parser.add_argument(
        '--solver', choices=['ppo', 'sac'], default=['sac'], nargs='+', help='ppo and/or sac'
    )
    args = parser.parse_args()

    # Generate main function for each combination of arguments
    processes = []
    queues = []
    for env in args.env:
        for solver in args.solver:
            main_func = get_main_function(env, solver)
            queue = multiprocessing.Queue()
            proc = multiprocessing.Process(target=wrapper, args=(main_func, queue))
            processes.append(proc)
            queues.append(queue)
            proc.start()

    # Join all processes
    for proc in processes:
        proc.join()

    # Collect results and generate plots
    for queue, env, solver in zip(queues, args.env*len(args.solver), args.solver*len(args.env)):
        result, model_path = queue.get()
        print("Generating plots")
        plot_reward_graph_multi(result, f"{env.upper()} {solver.upper()}")
    #
    # plt.show()
    # plt.close(1)


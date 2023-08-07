import multiprocessing

import matplotlib.pyplot as plt

from DI_PPO import main as main_ppo
from DI_SAC import main as main_sac
from utilities.plotting import plot_reward_graph_multi


def wrapper(func, queue):
    queue.put(func())


if __name__ == "__main__":
    queues = [multiprocessing.Queue() for _ in [main_ppo, main_sac]]
    procs = [multiprocessing.Process(target=wrapper, args=(func, q)) for func, q in zip([main_ppo, main_sac], queues)]

    [p.start() for p in procs]
    [p.join() for p in procs]

    ppo_res, sac_res = [q.get() for q in queues]

    plot_reward_graph_multi(ppo_res, "DI PPO")
    plot_reward_graph_multi(sac_res, "DI SAC")
    plt.show()

import torch
from torchdiffeq_ctrl import odeint_adjoint as odeint
from utilities.torch_device import device
import math

def bisection_search(func, criteria, lower_bound, upper_bound, max_iterations=7):
    iteration = 0
    update = False
    for _ in range(max_iterations):
        mid = int((lower_bound+upper_bound)/2)

        if mid == lower_bound:
            return lower_bound

        if criteria(func(mid)):
            lower_bound, upper_bound = mid, upper_bound
            update = True
        else:
            lower_bound, upper_bound = lower_bound, mid

        iteration += 1

    return lower_bound, update


def optimal_time(init_time, max_time, dt, loss_func, x_init, func, init_loss):

    max_iteration = int(math.log2(max_time - init_time))

    def func_wrap(time):
        time = torch.linspace(0, (time - 1) * dt, time).to(device)
        time_input = time.clone().reshape(time.shape[0], 1, 1, 1).repeat(1, x_init.shape[0], 1, 1).requires_grad_(True)
        y, dydt = odeint(func, x_init, time, method='euler', options=dict(step_size=dt))
        return loss_func(y, dydt, time_input)[0]

    def criteria(loss):
        return loss < init_loss

    with torch.no_grad():
        return bisection_search(func_wrap, criteria, init_time, max_time, max_iteration)

import random
from models import Cartpole, ModelParams
from neural_value_synthesis_diffeq import *
from torchdiffeq_ctrl import odeint_adjoint as odeint
from utilities.torch_device import device
from utilities.mujoco_torch import SimulationParams
from time_search import optimal_time
from PSDNets import ICNN
from utilities.mj_renderer import *
import wandb
import argparse
parser = argparse.ArgumentParser(description='Seed input')
parser.add_argument('--seed', type=int, default=4, help='Random seed')
args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)

wandb.init(project='CP_balancing', anonymous="allow")

sim_params = SimulationParams(6, 4, 2, 2, 1, 1, 100, 100, 0.008)
cp_params = ModelParams(2, 2, 1, 4, 4)
max_iter, max_time, alpha, dt, discount, step, scale, mode = 61, 146, .5, 0.008, 20, .005, 1, 'fwd'
Q = torch.diag(torch.Tensor([25, 25, 0.5, .1])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([0.001])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([25, 25, 0.5, .1])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime-0, sim_params.nsim, 1, 1))
cartpole = Cartpole(sim_params.nsim, cp_params, device, mode='fwd')
renderer = MjRenderer("./xmls/cartpole.xml", 0.0001)


def build_discounts(lambdas: torch.Tensor, discount: float):
    for i in range(lambdas.shape[0]):
        lambdas[i, :, :, :] *= (discount)**i

    return lambdas.clone()


def atan2_encoder(x: torch.Tensor):
    return torch.pi ** 2 * torch.sin(x)


def state_encoder(x: torch.Tensor):
    b, r, c = x.shape
    x = x.reshape((b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
    qp = atan2_encoder(qp)
    return torch.cat((qc, qp, v), 1).reshape((b, r, c))


def batch_state_encoder(x: torch.Tensor):
    t, b, r, c = x.shape
    x = x.reshape((t*b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
    qp = atan2_encoder(qp)
    return torch.cat((qc, qp, v), 1).reshape((t, b, r, c))


class NNValueFunction(nn.Module):
    def __init__(self, n_in):
        super(NNValueFunction, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(n_in+1, 200),
            nn.Softplus(beta=5),
            nn.Linear(200, 500),
            nn.Softplus(beta=5),
            nn.Linear(500, 1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.uniform_(m.bias)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        x = x.reshape(x.shape[0], sim_params.nqv)
        b = x.shape[0]
        time = torch.ones((b, 1)).to('cuda') * t
        aug_x = torch.cat((x, time), dim=1)
        return self.nn(aug_x).reshape(b, 1, 1)


nn_value_func = ICNN([sim_params.nqv+1, 200, 500, 1]).to(device)

def loss_func(x: torch.Tensor):
    x = state_encoder(x)
    return x @ Q @ x.mT


def loss_quadratic(x, gain):
    return x @ gain @ x.mT


def backup_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1].view(1, nsim, r, c).clone()
    x_init = x[0].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final).reshape(nsim, r, c)
    x_init_w = batch_state_encoder(x_init).reshape(nsim, r, c)
    value_final = nn_value_func(0, x_final_w).squeeze()
    value_init = nn_value_func((sim_params.ntime - 1) * dt, x_init_w).squeeze()

    return -value_init+value_final


def NSD_loss(x: torch.Tensor):
    _, nsim, r, c = x.shape
    zero = torch.zeros((nsim, r, c)).to(device)
    value_final = nn_value_func(0, zero).squeeze()
    value_init = nn_value_func((sim_params.ntime - 1) * dt, zero).squeeze()

    return -value_init+value_final


def batch_state_loss(x: torch.Tensor):
    x = batch_state_encoder(x)
    t, nsim, r, c = x.shape
    x_run = x[:-4].view(t-4, nsim, r, c).clone()
    x_final = x[-4:].view(4, nsim, r, c).clone()
    l_running = (loss_quadratic(x_run, Q)).squeeze()
    l_terminal = loss_quadratic(x_final, Qf).squeeze()

    return torch.cat((l_running, l_terminal), dim=0)


def value_terminal_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final).reshape(nsim, r, c)
    value_final = nn_value_func(0, x_final_w).squeeze()
    return value_final


def batch_ctrl_loss(acc: torch.Tensor):
    qddc = acc[:, :, :, 0].unsqueeze(2).clone()
    l_ctrl = torch.sum(qddc @ R @ qddc.mT, 0).squeeze()
    return torch.mean(l_ctrl)


def batch_inv_dynamics_loss(x, acc, alpha):
    x, acc = x[:-1].clone(), acc[:-1, :, :, sim_params.nv:].clone()
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q = q.reshape((q.shape[0]*q.shape[1], 1, sim_params.nq))
    x_reshape = x.reshape((x.shape[0]*x.shape[1], 1, sim_params.nqv))
    M = cartpole._Mfull(q).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    C = cartpole._Tbias(x_reshape).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    Tf = cartpole._Tfric(v).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    u_batch = ((M @ acc.mT).mT - C + Tf) * torch.Tensor([1, 0]).to(device)
    return u_batch @ torch.linalg.inv(M) @ u_batch.mT / scale


def loss_function(x, acc, alpha=1):
    l_run_ctrl = torch.sum(batch_inv_dynamics_loss(x, acc, alpha).squeeze(), dim=0)
    l_run_state = torch.sum(batch_state_loss(x).squeeze(), dim=0)
    l_run = l_run_state + l_run_ctrl
    l_bellman = backup_loss(x)
    l_terminal = 100 * value_terminal_loss(x)
    l_nsd = torch.mean(torch.square(NSD_loss(x))) * 100
    loss = torch.mean(l_run + l_bellman + l_terminal)
    return torch.maximum(loss, torch.zeros_like(loss)) + l_nsd*0

cartpole.GEAR = 1
cartpole.LENGTH = 0.3
cartpole.MASS_P = 0.1
cartpole.FRICTION = torch.Tensor([0.0, 0.1]).to(device)

dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=cartpole, mode=mode, step=step, scale=scale,
).to(device)
time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device)
one_step = torch.linspace(0, dt, 2).to(device)
optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=6e-3, amsgrad=True)

log = f"LYAP_CP_m-{mode}_d-{discount}_s-{step}_seed{seed}"


if __name__ == "__main__":
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    qc_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(-2, 2) * 1
    qp_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(-0.6, 0.6)
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(0.01, 0.01)
    x_init = torch.cat((qc_init, qp_init, qd_init), 2).to(device)
    iteration = 0
    alpha = 0

    while iteration < max_iter:
        optimizer.zero_grad()
        x_init = x_init[torch.randperm(sim_params.nsim)[:], :, :].clone()
        traj, dtrj_dt = odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=dt))
        loss = loss_function(traj, dtrj_dt, alpha)
        loss.backward()
        optimizer.step()
        sim_params.ntime, _ = optimal_time(sim_params.ntime, max_time, dt, loss_function, x_init, dyn_system, loss)

        optimizer.step()
        wandb.log({'epoch': iteration + 1, 'loss': loss.item()})

        print(f"Epochs: {iteration}, Loss: {loss.item()}\n")

        if iteration % 15 == 0:
            for i in range(0, sim_params.nsim, 20):
                selection = random.randint(0, sim_params.nsim - 1)
                renderer.render(traj[:, selection, 0, :sim_params.nq].cpu().detach().numpy())

        iteration += 1

    model_scripted = torch.save(dyn_system.value_func.to('cpu').state_dict(), f'{log}.pt')  # Export to TorchScript

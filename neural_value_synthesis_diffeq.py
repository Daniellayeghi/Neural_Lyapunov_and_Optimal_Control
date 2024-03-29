import torch
import torch.nn.functional as Func
from torch import nn
from utilities.torch_device import device
from utilities.mujoco_torch import SimulationParams

def decomp_x(x, sim_params: SimulationParams):
    return x[:, :, 0:sim_params.nq].clone(), x[:, :, sim_params.nq:].clone()


def decomp_xd(xd, sim_params: SimulationParams):
    return xd[:, :, 0:sim_params.nv].clone(), xd[:, :, sim_params.nv:].clone()


def compose_xxd(x, acc):
    return torch.cat((x, acc), dim=3)


def compose_acc(qd, dt):
    acc = torch.diff(qd, dim=0) / dt
    return acc


class ProjectedDynamicalSystem(nn.Module):
    def __init__(self, value_function, loss, sim_params: SimulationParams, dynamics=None, encoder=None, mode='proj', scale=1, step=1, R=None):
        super(ProjectedDynamicalSystem, self).__init__()
        self.value_func = value_function
        self.loss_func = loss
        self.sim_params = sim_params
        self.nsim = sim_params.nsim
        self._dynamics = dynamics
        self._encoder = encoder
        self._acc_buffer = torch.zeros((sim_params.ntime, sim_params.nsim, 1, 1)).to(device).requires_grad_(False)
        self._scale = scale
        self.step = step
        self._step_func = self._dynamics
        self._R = R
        self.collect = True
        self._policy = None

        if R is None:
            self._reg_func = lambda q: self._dynamics._M_reg(q)
        else:
            self._reg_func = lambda q: torch.inverse(R)

        if mode == 'proj':
            self._ctrl = self.project
        if mode == 'fwd':
            def underactuated_fwd_policy(q, v, x, Vx):
                # This function computes the HJB policy for force/torque based control
                # using the dynamics -> qdd = M_inv * (T_bias + Bu) where df/du = dxd/du and not dqdd/du
                dfdu_top = torch.zeros((sim_params.nsim, sim_params.nv, sim_params.nu)).to(device)
                M = self._dynamics._Mfull(q)
                Minv = torch.inverse(M)
                B = self._dynamics._Bvec()
                # TODO For some reason the -Minv which is wrong I think. At times performs better
                # dfdu = torch.cat((dfdu_top, (-Minv @ B.mT).mT), dim=1)
                # return -0.5 * self._scale * (M @ dfdu.mT @ Vx.mT).mT
                dfdu = torch.cat((dfdu_top, (Minv @ B.mT)), dim=1)
                reg = self._reg_func(q) * self._scale
                return -0.5 * self._scale * (reg @ dfdu.mT @ Vx.mT).mT

            self._policy = underactuated_fwd_policy
            self._ctrl = self.hjb
            self._step_func = self._dynamics

        if mode == 'inv':
            def underactuated_inv_policy(q, v, x, Vx):
                Vqd = Vx[:, :, self.sim_params.nq:].clone()

                M = self._dynamics._Mfull(q)
                Tb = self._dynamics._Tbias(x)
                Tf = self._dynamics._Tfric(v)
                ones = torch.ones((v.shape[0], 1, 1)).to(device)
                zeros = torch.zeros((v.shape[0], 1, 1)).to(device)
                Tbias = (Tb - Tf)

                Mu, Mua = self._dynamics._Mu_Mua(q)

                if Mu is None:
                    Minv = torch.linalg.inv(M)
                    dfdqd = torch.eye(self.sim_params.nv).repeat(self.sim_params.nsim, 1, 1).to(device)
                    return (Minv @ (Tbias - .5 * self._scale * Vqd @ dfdqd).mT).mT

                Tbiasc, Tbiasu = Tbias[:, :, 0].reshape(self.sim_params.nsim, 1, 1).clone(), Tbias[:, :, 1:].reshape(
                    self.sim_params.nsim, 1, self.sim_params.nv-1).clone()
                Mcc, Mcu, Muu = M[:, 0, 0].reshape(self.sim_params.nsim, 1, 1).clone(), M[:, 0, 1:].reshape(
                    self.sim_params.nsim, 1, self.sim_params.nv-1).clone(), M[:, 1:, 1:].reshape(self.sim_params.nsim, self.sim_params.nv-1, self.sim_params.nv-1).clone()
                Ba = torch.cat((ones, (-torch.inverse(Muu) @ Mcu.mT).mT), dim=2)
                Fm = torch.cat((zeros, (torch.inverse(Muu) @ Tbiasu.mT).mT), dim=2)
                return torch.inverse(Ba @ M @ Ba.mT) @ (Ba @ Tbias.mT - 0.5 * self._scale * Vqd @ Ba.mT - Ba @ M @ Fm.mT)


            self._policy = underactuated_inv_policy
            self._ctrl = self.hjb
            self._step_func = self._dynamics

        if dynamics is None:
            def dynamics(x, acc):
                v = x[:, :, self.sim_params.nq:].view(self.sim_params.nsim, 1, self.sim_params.nv).clone()
                return torch.cat((v, acc), 2)

            self._step_func = dynamics

            def policy(q, v, x, Vqd):
                return self._scale * -0.5 * Vqd

            self._policy = policy

    def hjb(self, t, x):
        x_enc = self._encoder(x)
        q, v = decomp_x(x, self.sim_params)

        def dvdx(t, x, value_net):
            with torch.set_grad_enabled(True):
                x = x.detach().requires_grad_(True)
                value = value_net(t, x).requires_grad_()
                dvdx = torch.autograd.grad(
                    value, x, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
                )[0]
                return dvdx

        Vx = dvdx(t, x_enc, self.value_func)
        return self._policy(q, v, x, Vx)

    def project(self, t, x):
        x_enc = self._encoder(x)
        q, v = decomp_x(x, self.sim_params)
        xd = torch.cat((v, torch.zeros_like(v)), 2)

        def dvdt(t, x, value_net):
            with torch.set_grad_enabled(True):
                time = t.detach().requires_grad_(True)
                x = x.detach().requires_grad_(True)
                value = value_net(time, x).requires_grad_()
                dvdt = torch.autograd.grad(
                    value, time, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
                )[0]
                return dvdt

        def dvdx(t, x, value_net):
            with torch.set_grad_enabled(True):
                x = x.detach().requires_grad_(True)
                value = value_net(t, x).requires_grad_()
                dvdx = torch.autograd.grad(
                    value, x, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
                )[0]
                return dvdx

        Vx = dvdx(t, x_enc, self.value_func)
        # Vt = dvdt(t, x_enc, self.value_func)
        norm = ((Vx @ Vx.mT) + 1e-6).sqrt().view(self.nsim, 1, 1)
        unnorm_porj = Func.relu((Vx @ xd.mT) + self.step * self.loss_func(x))
        xd_trans = - (Vx / norm) * unnorm_porj
        return torch.clamp(xd_trans[:, :, self.sim_params.nv:].view(self.sim_params.nsim, 1, self.sim_params.nv), -40, 40)



    def dfdt(self, t, x):
        # TODO: Either the value function is a function of just the actuation space e.g. the cart or it takes into
        # TODO: the main difference is that the normalised projection is changed depending on what is used
        acc = self._ctrl(t, x)
        return self._dynamics(x, acc)


    def forward(self, t, x):
        u = self._ctrl(t, x)
        xd = self._step_func(x, u)
        return xd

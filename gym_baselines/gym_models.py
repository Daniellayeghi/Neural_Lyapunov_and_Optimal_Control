import gymnasium as gym
import numpy as np
from typing import Optional

import torch.linalg
from gymnasium import spaces
from gymnasium.utils import seeding
from utilities.gym_utils import PolicyVisualizer


class CustomEnv(gym.Env):
    def __init__(self, env_id, init_bound, terminal_time, observation_dim, action_dim, action_bounds):
        super(CustomEnv, self).__init__()

        self._iter = 0
        self.init_bound = init_bound
        self._terminal_time = terminal_time
        self._env_id = env_id
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-action_bounds, high=action_bounds, shape=(action_dim,), dtype=np.float32)
        self.transform_func = None

    def _terminated(self):
        return self._iter >= self._terminal_time

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self._iter = 0
        if seed is not None:
            self.seed(seed)

        # Validate that the bounds are correctly specified for each joint
        if len(self.init_bound['position']) != len(self.init_bound['velocity']):
            raise ValueError("Mismatch between the number of position and velocity bounds")

        num_joints = len(self.init_bound['position'])

        # Generate initial state based on the given bounds for position and velocity for each joint
        position_init = np.array([
            self.np_random.uniform(low=low, high=high)
            for (low, high) in self.init_bound['position']
        ])

        velocity_init = np.array([
            self.np_random.uniform(low=low, high=high)
            for (low, high) in self.init_bound['velocity']
        ])

        # Combine the initial position and velocity states
        self._init_state = np.concatenate([position_init, velocity_init])
        self.state = self._init_state

        if np.isnan(self.state).any():
            print("The state contains NaN values!")
        return self.state, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class CustomDoubleIntegrator(CustomEnv):
    def __init__(self, env_id='custom_cp', init_bound=(-np.inf, np.inf), terminal_time=100):
        super(CustomDoubleIntegrator, self).__init__(env_id, init_bound, terminal_time, 2, 1, 10)

        self._friction = .1
        self._mass = 1
        self._gear = 1
        self._dt = 0.01
        self._Q = np.diag(np.array([10, .1]))
        self._Qf = np.diag(np.array([10, .1])) * 10
        self._R = np.array([[1]])

    def _get_reward(self, state, u):
        Q = self._Q
        if self._iter >= self._terminal_time - 4:
            Q = self._Qf
        return -1*(state.T @ Q @ state + u.T @ self._R @ u)

    def _enc_state(self):
        q, qd = self.state
        return np.array([q, qd], dtype=np.float32)

    def step(self, u):
        q, qd = self.state
        qdd = (1 / self._mass * (u * self._gear - qd * self._friction))[0]

        q_new = q + qd * self._dt
        qd_new = qd + qdd * self._dt
        self.reward = self._get_reward(self._enc_state(), u)
        self.state = np.array([q_new, qd_new]).flatten()
        self._iter += 1

        terminate = self._terminated()

        return self.state, self.reward, terminate, terminate, {}


class CustomReacher(CustomEnv):
    def __init__(self, env_id, init_bound, terminal_time):
        super(CustomReacher, self).__init__(env_id, init_bound, terminal_time, 4, 2, 3)

        self._friction = np.array([0.025, 0.025]).reshape(2, 1)
        self._B = np.array([1, 1]).reshape(2, 1)
        self._gear = 0, 1
        self._dt = 0.01
        self._Q = np.diag(np.array([1, 1, 0, 0]))
        self._Qf = np.diag(np.array([100, 100, 1, 1]))
        self._R = np.diag(np.array([1, 1]))

        def transform_func(traj: np.array):
            traj[..., 1] = np.pi - (traj[..., 0] + (np.pi - traj[..., 1]))
            return traj

        self.transform_func = transform_func

    def _get_reward(self, state, u):
        Q = self._Q
        if self._iter >= self._terminal_time - 10:
            Q = self._Qf

        return -1*(state.T @ Q @ state + u.T @ self._R @ u)

    def _enc_state(self):
        q1, q2, qd1, qd2 = self.state
        enc = lambda x: np.arctan2(np.sin(x), np.cos(x))
        return np.array([enc(q1), enc(q2), qd1, qd2], dtype=np.float32)

    def step(self, u):
        q1, q2, qd1, qd2 = self.state
        qd = np.array([qd1, qd2]).reshape(2, 1)
        a1, a2, a3 = 0.025 + 0.045 + 1.4 * 0.3 ** 2, 0.3 * 0.16, 0.045
        M11 = a1 + 2 * a2 * np.cos(q2)
        M12 = a3 + a2 * np.cos(q2)
        M22 = a3

        M = np.array([M11, M12, M12, M22]).reshape(2, 2)
        a4 = 0.3 * 0.1
        T1act = -qd2 * (2 * qd1 + qd2)
        T2act = qd1 ** 2
        Tbias = np.array([T1act, T2act]).reshape(2, 1)
        Tbias = -Tbias * (a4 * np.sin(q2))
        Tfric = self._friction * qd

        qdd = np.linalg.pinv(M) @ (Tbias + u.reshape(2, 1) * self._B - Tfric).flatten()
        qdd1, qdd2 = qdd[0], qdd[1]

        qc_new = q1 + qd1 * self._dt
        qdc_new = qd1 + qdd1 * self._dt
        qp_new = q2 + qd2 * self._dt
        qdp_new = qd2 + qdd2 * self._dt
        self.reward = self._get_reward(self._enc_state(), u)
        self.state = np.array([qc_new, qp_new, qdc_new, qdp_new]).flatten()
        self._iter += 1

        terminate = self._terminated()
        return self._enc_state(), self.reward, terminate, terminate, {}


class CustomCartpole(CustomEnv):
    def __init__(self, env_id, init_bound=(-np.inf, np.inf), terminal_time=100, return_state=False):
        super(CustomCartpole, self).__init__(
            env_id, init_bound, terminal_time, 4, 1, 500
        )
        # Parameters specific to the cart-pole environment
        self._mass_p, self._mass_c, self._l = 1, 1, 1
        self._g, self._gear = -9.81, 1
        self._fr = np.array([0, .1]).reshape(2, 1)
        self._Q = np.diag(np.array([0, 0, 0, .0]))
        self._Qf = np.diag(np.array([80, 600, 0.8, 4.5]))
        self._R = np.array([[0.5]])/10
        self._dt = .01
        self.retrun_state = return_state
        self.state = np.zeros(4)

    def _get_reward(self, state, u):
        Q = self._Q
        if self._iter >= self._terminal_time - 10:
            Q = self._Qf

        return -(state.T @ Q @ state + u.T @ self._R @ u)

    def _enc_state(self):
        qc, qp, qdc, qdp = self.state
        enc = lambda x: (1 - np.cos(x)) / 2
        return np.array([qc, enc(qp), qdc, qdp], dtype=np.float32)

    def _get_mass_matrix(self, state):
        qc, qp, qdc, qdp = self.state
        m_p, m_c, g, gear, l = self._mass_p, self._mass_c, self._g, self._gear, self._l
        M = np.array([m_p + m_c, m_p * l * np.cos(qp), m_p * l * np.cos(qp), m_p * l ** 2]).reshape(2, 2)
        return M

    def step(self, u):
        # Something is wrong in the computing the effective torque on the pole here
        # limit force applied. Seems like at very high forces the behaviour is completely linear
        
        qc, qp, qdc, qdp = self.state
        qd = np.array([qdc, qdp]).reshape(2, 1)
        m_p, m_c, g, gear, l = self._mass_p, self._mass_c, self._g, self._gear, self._l
        M = np.array([m_p + m_c, m_p * l * np.cos(qp), m_p * l * np.cos(qp), m_p * l ** 2]).reshape(2, 2)
        C = np.array([0, -m_p * l * qdp * np.sin(qp), 0, 0]).reshape(2, 2)
        Tg = np.array([0, -m_p * g * l * np.sin(qp)]).reshape(2, 1)
        B = np.array([1, 0]).reshape(2, 1)
        self._reg = np.linalg.inv(np.array([[M[0, 0]]]))

        qdd = (np.linalg.inv(M) @ (-C @ qd - self._fr * qd + Tg + B * u)).flatten()
        qddc, qddp = qdd[0], qdd[1]
        qc_new = qc + qdc * self._dt
        qdc_new = qdc + qddc * self._dt
        qp_new = qp + qdp * self._dt
        qdp_new = qdp + qddp * self._dt
        self.reward = self._get_reward(self._enc_state(), u)
        self.state = np.array([qc_new, qp_new, qdc_new, qdp_new]).flatten()
        self._iter += 1

        terminate = self._terminated()

        # if terminate:
        #     self.reward *= 1000

        return self.state, self.reward, terminate, terminate, {}

    def _terminated(self):
        # out_of_bounds = np.abs(self.state[1]) > 0.75
        return self._iter >= self._terminal_time


class CustomCartpoleBalance(CustomEnv):
    def __init__(self, env_id, init_bound=(-np.inf, np.inf), terminal_time=100, return_state=False):
        super(CustomCartpoleBalance, self).__init__(
            env_id, init_bound, terminal_time, 4, 1, 30
        )
        # Parameters specific to the cart-pole environment
        self._mass_p, self._mass_c, self._l = 1, .1, .3
        self._g, self._gear = -9.81, 1
        self._fr = np.array([.1, .1]).reshape(2, 1)
        self._Q = np.diag(np.array([0, 25, 0.5, .1]))
        self._Qf = np.diag(np.array([0, 100, 0.5, 1]))
        self._R = np.array([[0.9]])
        self._dt = .01
        self.retrun_state = return_state
        self.state = np.zeros(4)

    def _get_reward(self, state, u):
        Q = self._Q
        if self._iter >= self._terminal_time - 4:
            Q = self._Qf

        return -(state.T @ Q @ state + u.T @ self._R @ u)

    def _enc_state(self):
        qc, qp, qdc, qdp = self.state
        enc = lambda x: np.pi ** 2 * np.sin(x)
        return np.array([qc, enc(qp), qdc, qdp], dtype=np.float32)

    def _get_mass_matrix(self, state):
        qc, qp, qdc, qdp = self.state
        m_p, m_c, g, gear, l = self._mass_p, self._mass_c, self._g, self._gear, self._l
        M = np.array([m_p + m_c, m_p * l * np.cos(qp), m_p * l * np.cos(qp), m_p * l ** 2]).reshape(2, 2)
        return M
    def step(self, u):
        qc, qp, qdc, qdp = self.state
        qd = np.array([qdc, qdp]).reshape(2, 1)
        m_p, m_c, g, gear, l = self._mass_p, self._mass_c, self._g, self._gear, self._l
        M = np.array([m_p + m_c, m_p * l * np.cos(qp), m_p * l * np.cos(qp), m_p * l ** 2]).reshape(2, 2)
        C = np.array([0, -m_p * l * qdp * np.sin(qp), 0, 0]).reshape(2, 2)
        Tg = np.array([0, -m_p * g * l * np.sin(qp)]).reshape(2, 1)
        B = np.array([1, 0]).reshape(2, 1)
        self._reg = np.linalg.inv(np.array([[M[0, 0]]]))

        qdd = (np.linalg.inv(M) @ (-C @ qd - self._fr * qd + Tg + B * u)).flatten()
        qddc, qddp = qdd[0], qdd[1]
        qc_new = qc + qdc * self._dt
        qdc_new = qdc + qddc * self._dt
        qp_new = qp + qdp * self._dt
        qdp_new = qdp + qddp * self._dt
        self.reward = self._get_reward(self._enc_state(), u)
        self.state = np.array([qc_new, qp_new, qdc_new, qdp_new]).flatten()
        self._iter += 1

        terminate = self._terminated()

        # if terminate:
        #     self.reward *= 1000

        return self.state, self.reward, terminate, terminate, {}

    def _terminated(self):
        # out_of_bounds = np.abs(self.state[1]) > 0.75
        return self._iter >= self._terminal_time
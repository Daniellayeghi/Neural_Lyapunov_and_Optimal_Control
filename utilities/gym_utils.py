import torch
import numpy as np
from utilities.mj_renderer import MjRenderer



class PolicyVisualizer:
    def __init__(self, model, env, xml_path: str, transform_func=lambda x: x):

        self._model = model
        self._env = env
        self._renderer = MjRenderer(xml_path)
        self._transform_func = transform_func

    def visualize(self, horizon: int = 1000) -> None:
        try:
            while True:  # Keep running until interrupted
                obs, _ = self._env.reset()
                positions = []

                for _ in range(horizon):
                    action, _ = self._model.predict(obs)
                    obs, _, _, _, _ = self._env.step(action)
                    pos_size = obs.shape[0]//2
                    positions.append(obs[:pos_size])

                self._renderer.render(np.array(positions))

        except KeyboardInterrupt:  # Handle interruption
            print("Visualization stopped.")
            return


class ValueFunction:

    def __init__(self, model):
        self._model = model

    def __call__(self, state):
        """
        Compute the state value for a given state using the forward method.

        :param model: The trained SB3 model with the provided forward method.
        :param state: The state for which the value should be computed.
        :return: The state value V(s).
        """
        # Convert the state to a PyTorch tensor
        state_tensor = torch.as_tensor(state, device=self._model.policy.device).unsqueeze(0)

        # Use the forward method to get the action, value, and log probability
        _, values, _ = self._model.policy.forward(state_tensor, deterministic=True)

        # Extract the scalar value
        state_value = values.item()

        return state_value



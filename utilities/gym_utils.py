import numpy as np
from utilities.mj_renderer import MjRenderer


class PolicyVisualizer:
    def __init__(self, model, env, xml_path: str):

        self.model = model
        self.env = env
        self.renderer = MjRenderer(xml_path)

    def visualize(self, horizon: int = 1000) -> None:
        obs, _ = self.env.reset()
        positions = []

        for _ in range(horizon):
            action, _ = self.model.predict(obs)
            obs, _, _, _, _ = self.env.step(action)
            # Assuming the first element of obs is the position
            positions.append(obs[0])

        # Render the sequence of positions
        self.renderer.render(np.array(positions))


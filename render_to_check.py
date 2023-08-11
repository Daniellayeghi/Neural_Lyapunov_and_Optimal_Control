from gym_baselines.gym_models import CustomCartpole
from utilities.gym_utils import PolicyVisualizer


init_bound = {
    'position': [-.1, .1],  # Upright position
    'velocity': [0, 0]   # No initial velocity
}
terminal_time = 300
env = CustomCartpole(
    env_id='custom_cartpole', init_bound=init_bound, terminal_time=terminal_time, return_state=True
)


class ZeroControlModel:
    def predict(self, obs):
        return 0, None  # Always returning 0 control input and None for state


model = ZeroControlModel()
xml_path = "xmls/cartpole.xml"
visualizer = PolicyVisualizer(model, env, xml_path)
visualizer.visualize(horizon=terminal_time)

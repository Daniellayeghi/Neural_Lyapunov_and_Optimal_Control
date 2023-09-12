import torch.nn as nn
from stable_baselines3.common.utils import get_linear_fn
from gym_baselines.gym_models import *

initial_clip_range = 1.0
final_clip_range = 0.0
end_fraction = 0.4

clip_range_schedule = get_linear_fn(initial_clip_range, final_clip_range, end_fraction)

# Configurations for each environment and solver combination
configurations = {
    'DI_PPO': {
        'model_type': CustomDoubleIntegrator,
        'env_name': 'Custom Double Integrator PPO',
        'epochs': 417,
        'terminal_time': 400,
        'nproc': 6,
        'hyperparameters': {
            'ent_coef': 0.0,
            'gae_lambda': 0.98,
            'gamma': 0.99,
            'n_epochs': 4,
            'n_steps': 16,
            'policy': 'MlpPolicy',
            'normalize_advantage': True,
            'tensorboard_log': './ppo_tensorboard_di/',
            'verbose': 1
        },
        'model_params': {
            'env_id': 'Custom Double Integrator PPO',
            'init_bound': {
                'position': [(-3, 3)],
                'velocity': [(-3, 3)]
            },
            'terminal_time': 400
        },
        'xml_path': './xmls/doubleintegrator.xml'
    },
    'DI_SAC': {
        'model_type': CustomDoubleIntegrator,
        'env_name': 'Custom Double Integrator SAC',
        'epochs': 417,
        'terminal_time': 400,
        'nproc': 6,
        'hyperparameters': {
            'policy': 'MlpPolicy',
            'batch_size': 512,
            'buffer_size': 50000,
            'ent_coef': 0.1,
            'gamma': 0.9999,
            'gradient_steps': 32,
            'learning_rate': 0.0003,
            'learning_starts': 0,
            'tau': 0.01,
            'train_freq': 32,
            'use_sde': True,
            'policy_kwargs': {
                'log_std_init': -3.67,
                'net_arch': [64, 64]
            },
            'tensorboard_log': './sac_tensorboard_di/',
            'verbose': 1
        },
        'model_params': {
            'env_id': 'Custom Double Integrator SAC',
            'init_bound': {
                'position': [(-3, 3)],
                'velocity': [(-3, 3)]
            },
            'terminal_time': 400
        },
        'xml_path': './xmls/doubleintegrator.xml'
    },
    'TL_PPO': {
        'model_type': CustomReacher,
        'env_name': 'Custom Reacher PPO',
        'epochs': 975,
        'terminal_time': 171,
        'nproc': 6,
        'hyperparameters': {
            'batch_size': 64,
            'clip_range': 0.4,  # Placeholder, since this uses a function
            'ent_coef': 0.0,
            'gae_lambda': 0.9,
            'gamma': 0.99,
            'learning_rate': 3e-05,
            'max_grad_norm': 0.5,
            'n_epochs': 20,
            'n_steps': 512,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'log_std_init': -2.7,
                'ortho_init': False,
                'activation_fn': nn.ReLU,
                'net_arch': {
                    'pi': [256, 256],
                    'vf': [256, 256]
                }
            },
            'sde_sample_freq': 4,
            'use_sde': True,
            'vf_coef': 0.5,
            'normalize_advantage': True,
            'tensorboard_log': './ppo_tensorboard_tl/',
            'verbose': 1
        },
        'model_params': {
            'env_id': 'Custom Reacher PPO',
            'init_bound': {
                'position': [(-3, 3), (-3, 3)],  # bounds for joint 1, joint 2, joint 3
                'velocity': [(0, 0), (0, 0)]  # bounds for joint 1, joint 2, joint 3
            },
            'terminal_time': 171
        },
        'xml_path': './xmls/reacher.xml'
    },
    'TL_SAC': {
        'model_type':CustomReacher,
        'env_name': 'Custom Reacher SAC',
        'epochs': 300,
        'terminal_time': 171,
        'nproc': 6,
        'hyperparameters': {
            'batch_size': 256,
            'buffer_size': 300000,
            'ent_coef': 'auto',
            'gamma': 0.98,
            'gradient_steps': 64,
            'learning_rate': 0.00073,
            'learning_starts': 10000,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'log_std_init': -3,
                'net_arch': [400, 300]
            },
            'tau': 0.02,
            'train_freq': 64,
            'use_sde': True,
            'tensorboard_log': './sac_tensorboard_tl/',
            'verbose': 1
        },
        'model_params': {
            'env_id': 'Custom Reacher SAC',
            'init_bound': {
                'position': [(-3, 3), (-3, 3)],  # bounds for joint 1, joint 2, joint 3
                'velocity': [(0, 0), (0, 0)]  # bounds for joint 1, joint 2, joint 3
            },
            'terminal_time': 171
        },
        'xml_path': './xmls/reacher.xml'
    },
    'CP_BALANCE_PPO': {
        'model_type': CustomCartpoleBalance,
        'env_name': 'Custom Cartpole Balance PPO',
        'epochs': 1500,
        'terminal_time': 110,
        'nproc': 6,
        'hyperparameters': {
            'batch_size': 128,
            'clip_range': 0.4,  # Placeholder, since this uses a function
            'ent_coef': 0.0,
            'gae_lambda': 0.92,
            'gamma': 0.99,
            'learning_rate': 3e-5,
            'max_grad_norm': 0.5,
            'n_epochs': 20,
            'n_steps': 512,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'log_std_init': -2.7,
                'activation_fn': nn.ReLU,
                'ortho_init': False,
                'net_arch': {
                    'pi': [256, 256],
                    'vf': [256, 256]
                }
            },
            'sde_sample_freq': 4,
            'use_sde': True,
            'vf_coef': 0.5,
            'normalize_advantage': True,
            'tensorboard_log': './ppo_tensorboard_cp_balance/',
            'verbose': 1
        },
        'model_params': {
            'env_id': 'Custom Cartpole Balance PPO',
            'init_bound': {
                'position': [(-2, 2), (-.6, .6)],  # bounds for joint 1, joint 2, joint 3
                'velocity': [(0.2, 0.2), (0.2, 0.2)]  # bounds for joint 1, joint 2, joint 3
            },
            'terminal_time': 110
        },
        'xml_path': './xmls/cartpole.xml'
    },
    'CP_BALANCE_SAC': {
        'xml_path': './xmls/cartpole.xml',
        'model_type': CustomCartpoleBalance,
        'env_name': 'Custom Cartpole Balance SAC',
        'epochs': 1500,
        'terminal_time': 110,
        'nproc': 6,
        'hyperparameters': {
            'batch_size': 256,
            'buffer_size': 300000,
            'ent_coef': 'auto',
            'gamma': 0.98,
            'gradient_steps': 64,
            'learning_rate': .00073,
            'learning_starts': 10000,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'activation_fn': nn.ReLU,
                'log_std_init': -3,
                'net_arch': [400, 300]
            },
            'tau': 0.02,
            'train_freq': 64,
            'use_sde': False,
            'tensorboard_log': './sac_tensorboard_cp_balance/',
            'verbose': 1
        },
        'model_params': {
            'env_id': 'Custom Cartpole Balance SAC',
            'init_bound': {
                'position': [(-2, 2), (-.6, .6)],  # bounds for joint 1, joint 2, joint 3
                'velocity': [(0.2, 0.2), (0.2, 0.2)]  # bounds for joint 1, joint 2, joint 3
            },
            'terminal_time': 110
        }
    },
    'CP_SWINGUP_SAC': {
        'xml_path': './xmls/cartpole.xml',
        'model_type': CustomCartpole,
        'env_name': 'Custom Cartpole Swingup SAC',
        'epochs': 4333,
        'terminal_time': 171,
        'nproc': 6,
        'hyperparameters': {
            'batch_size': 256,
            'buffer_size': 300000,
            'ent_coef': 'auto',
            'gamma': 0.98,
            'gradient_steps': 64,
            'learning_rate': .00073,
            'learning_starts': 10000,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'activation_fn': nn.ReLU,
                'log_std_init': -3,
                'net_arch': [400, 300]
            },
            'tau': 0.02,
            'train_freq': 64,
            'use_sde': False,
            'tensorboard_log': './sac_tensorboard_cp_swingup/',
            'verbose': 1
        },
        'model_params': {
            'env_id': 'Custom Cartpole Swingup SAC',
            'init_bound': {
                'position': [(0, 0), (2.84, 3.44)],  # bounds for joint 1, joint 2, joint 3
                'velocity': [(0, 0), (0, 0)]  # bounds for joint 1, joint 2, joint 3
            },
            'terminal_time': 171
        }
    },
    'CP_SWINGUP_PPO': {
        'xml_path': './xmls/cartpole.xml',
        'model_type': CustomCartpole,
        'env_name': 'Custom Cartpole Swingup PPO',
        'epochs': 4333,
        'terminal_time': 171,
        'nproc': 6,
        'hyperparameters': {
            'batch_size': 128,
            'clip_range': 0.4,  # Placeholder, since this uses a function
            'ent_coef': 0.0,
            'gae_lambda': 0.92,
            'gamma': 0.99,
            'learning_rate': 3e-5,
            'max_grad_norm': 0.5,
            'n_epochs': 20,
            'n_steps': 512,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'log_std_init': -2.7,
                'activation_fn': nn.ReLU,
                'ortho_init': False,
                'net_arch': {
                    'pi': [400, 300],
                    'vf': [200, 300]
                }
            },
            'sde_sample_freq': 4,
            'use_sde': True,
            'vf_coef': 0.5,
            'normalize_advantage': True,
            'tensorboard_log': './ppo_tensorboard_cp_swingup/',
            'verbose': 1
        },
        'model_params': {
            'env_id': 'Custom Cartpole Swingup PPO',
            'init_bound': {
                'position': [(0, 0), (2.84, 3.44)],  # bounds for joint 1, joint 2, joint 3
                'velocity': [(0, 0), (0, 0)]  # bounds for joint 1, joint 2, joint 3
            },
            'terminal_time': 171
        }
    }
}



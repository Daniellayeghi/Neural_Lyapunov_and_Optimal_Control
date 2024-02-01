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
        'epochs': 100,
        'terminal_time': 400,
        'nproc': 12,
        'hyperparameters': {
            'batch_size': 256,
            'clip_range': 0.4,  # Placeholder, since this uses a function
            'ent_coef': 0.0,
            'gae_lambda': 0.95,
            'gamma': 0.99,
            'learning_rate': 3e-05,
            'max_grad_norm': 0.5,
            'n_epochs': 20,
            'n_steps': 10,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                # 'log_std_init': -2.7,
                # 'ortho_init': False,
                'activation_fn': nn.ReLU,
                'net_arch': {
                    'pi': [64, 64],
                    'vf': [64, 64],
                }
            },
            'sde_sample_freq': 4,
            'use_sde': True,
            'vf_coef': 0.5,
            'normalize_advantage': True,
            'tensorboard_log': './ppo_tensorboard_di/',
            'verbose': 1
        },
        'model_params': {
            'env_id': 'Custom Double Integrator PPO',
            'init_bound': {
                'position': [(-1, 1)],
                'velocity': [(-.7, .7)]
            },
            'terminal_time': 400
        },
        'xml_path': './xmls/doubleintegrator.xml'
    },
    'DI_SAC': {
        'model_type': CustomDoubleIntegrator,
        'env_name': 'Custom Double Integrator SAC',
        'epochs': 100,
        'terminal_time': 400,
        'nproc': 12,
        'hyperparameters': {
            'batch_size': 256,
            'buffer_size': 1000000,
            'ent_coef': 'auto',
            'gamma': 0.98,
            'gradient_steps': 1,
            'learning_rate': 3e-4,
            'learning_starts': 1,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'net_arch': [64, 64]
            },
            'tau': 0.02,
            'train_freq': 1,
            'use_sde': True,
            'tensorboard_log': './sac_tensorboard_di/',
            'verbose': 1
        },
        'model_params': {
            'env_id': 'Custom Double Integrator SAC',
            'init_bound': {
                'position': [(-1, 1)],
                'velocity': [(-.7, .7)]
            },
            'terminal_time': 400
        },
        'xml_path': './xmls/doubleintegrator.xml'
    },
    'TL_PPO': {
        'model_type': CustomReacher,
        'env_name': 'Custom Reacher PPO',
        'epochs': 60,
        'terminal_time': 170,
        'nproc': 12,
        'hyperparameters': {
            'batch_size': 256,
            'clip_range': 0.4,  # Placeholder, since this uses a function
            'ent_coef': 0.0,
            'gae_lambda': 0.95,
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'max_grad_norm': 0.5,
            'n_epochs': 20,
            'n_steps': 1,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'activation_fn': nn.ReLU,
                'net_arch': {
                    'pi': [128, 128],
                    'vf': [128, 128]
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
        'epochs': 60,
        'terminal_time': 170,
        'nproc': 12,
        'hyperparameters': {
            'batch_size': 256,
            'buffer_size': 300000,
            'ent_coef': 'auto',
            'gamma': 0.98,
            'gradient_steps': 1,
            'learning_rate': 3e-4,
            'learning_starts': 1,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'net_arch': [128, 128]
            },
            'tau': 0.02,
            'train_freq': 1,
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
        'epochs': 100,
        'terminal_time': 79,
        'nproc': 12,
        'hyperparameters': {
            'batch_size': 256,
            'clip_range': 0.4,  # Placeholder, since this uses a function
            'ent_coef': 0.0,
            'gae_lambda': 0.95,
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'max_grad_norm': 0.5,
            'n_epochs': 20,
            'n_steps': 1,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'activation_fn': nn.ReLU,
                'net_arch': {
                    'pi': [128, 128],
                    'vf': [128, 128]
                }
            },
            'sde_sample_freq': 4,
            'use_sde': True,
            'vf_coef': 0.5,
            'normalize_advantage': True,
            'tensorboard_log': './ppo_tensorboard_balance/',
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
        'epochs': 100,
        'terminal_time': 79,
        'nproc': 12,
        'hyperparameters': {
            'batch_size': 256,
            'buffer_size': 1000000,
            'ent_coef': 'auto',
            'gamma': 0.98,
            'gradient_steps': 1,
            'learning_rate': 3e-4,
            'learning_starts': 1,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'net_arch': [128, 128],
            },
            'tau': 0.02,
            'train_freq': 1,
            'use_sde': True,
            'tensorboard_log': './sac_tensorboard_balance/',
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
        'epochs': 130,
        'terminal_time': 171,
        'nproc': 12,
        'hyperparameters': {
            'batch_size': 256,
            'buffer_size': 300000,
            'ent_coef': 'auto',
            'gamma': 0.98,
            'gradient_steps': 1,
            'learning_rate': 3e-4,
            'learning_starts': 1,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'activation_fn': nn.ReLU,
                'net_arch': [128, 128]
            },
            'tau': 0.02,
            'train_freq': 1,
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
        'epochs': 130,
        'terminal_time': 171,
        'nproc': 12,
        'hyperparameters': {
            'batch_size': 256,
            'clip_range': 0.4,  # Placeholder, since this uses a function
            'ent_coef': 0.0,
            'gae_lambda': 0.92,
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'max_grad_norm': 0.5,
            'n_epochs': 20,
            'n_steps': 1,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'log_std_init': -2.7,
                'activation_fn': nn.ReLU,
                'ortho_init': False,
                'net_arch': {
                    'pi': [128, 128],
                    'vf': [128, 128]
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


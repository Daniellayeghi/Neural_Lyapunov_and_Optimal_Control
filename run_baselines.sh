#!/bin/bash

# Run gym_gen_baselines.py with various environments and solvers
python3 gym_gen_baselines.py --env cp_swingup --solver ppo sac
python3 gym_gen_baselines.py --env cp_balance --solver ppo sac
python3 gym_gen_baselines.py --env di --solver ppo sac

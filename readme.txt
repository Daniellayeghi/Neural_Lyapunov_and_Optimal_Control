Implementation for Neural Lyapunov and Optimal Control

https://arxiv.org/abs/2305.15244

Execution steps:

This repository uses mujoco-py an therefore requires mujoco binaries. The steps for installing and copying
these binaries are mentioned in:

https://github.com/openai/mujoco-py

Once the "Install Mujoco" step is completed. First the user must install pipenv

- pip install pipenv

Then pipenv will install all necessary packages. Move to the cloned directory and use:

- pipenv install

A pipenv shell can then be created

- pipenv shell

To enable rendering update LD_LIBRARY_PATH in your current shell:

- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path_to_mujoco/mujoco210/bin

Each of the examples CP_balancing, CP_swingup, DI_lyapunov, TL_control can then be ran using a seed eg:

- python3 CP_balancing.py --seed 4

The training results will be plotted using Weights and Biases in anonymous mode. A link to this dashboard will be available once the above command is ran.

# Neural Lyapunov and Optimal Control

This repository uses `mujoco-py` and therefore requires mujoco binaries. The steps for installing and copying these binaries are mentioned in:
[https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py)

Once the "Install Mujoco" step is completed, first the user must install pipenv:

``
pip install pipenv
``

Then, pipenv will install all necessary packages. Move to the cloned directory and use:

``
pipenv install
``

``
pipenv run pip install torch==1.10.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113/
``

A pipenv shell can then be created:

``
pipenv shell
``

To enable rendering, update LD_LIBRARY_PATH in your current shell:

``
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path_to_mujoco/mujoco210/bin
``

Each of the examples CP_balancing, CP_swingup, DI_lyapunov, TL_control can then be ran using a seed, e.g.:

``
python3 CP_balancing.py --seed 4
``

The training results will be plotted using Weights and Biases in anonymous mode. A link to this dashboard will be available once the above command is ran.

Additionally, the baselines can be ran via:

``
python3 gym_gen_baselines.py --env [tl cp_balance cp_swingup di] --solver [ppo sac]
``

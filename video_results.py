import matplotlib
import mujoco_py
import numpy as np
from collections import namedtuple
from time import sleep
import os
os.environ['MUJOCO_PY_FORCE_EGL'] = 'True'


def custom_sine(x, amp, base_period=2 * np.pi):
    base_sine = amp * np.sin(x)
    additional_sine = amp * np.sin(x) if x >= base_period / 2 else 0
    return base_sine + additional_sine



if __name__ == "__main__":
    # generate sinosoidal trajectory in control space between 0.1 and -0.1 for 1000 timesteps
    # make frequency an adjustable parameter
    arm2_scale_lf = 0.005
    arm1_scale_lf = 0.075
    arm2_scale = 0.1
    arm1_scale = 0.2
    freq = 8
    timesteps = 1000
    time = np.linspace(0, 2 * np.pi * freq, timesteps)
    us = np.array([custom_sine(x, arm1_scale_lf) for x in time])
    import matplotlib.pyplot as plt
    plt.plot(us)
    plt.show()
    # us = arm2_scale * np.sin(time)

    mj_model = mujoco_py.load_model_from_path("./xmls/reacher_lf.xml")
    mj_sim = mujoco_py.MjSim(mj_model)
    mj_viewer = mujoco_py.MjViewer(mj_sim)
    State = namedtuple('State', 'time qpos qvel act udd_state')
    set_state = lambda pos_t: State(
        time=0, qpos=pos_t, qvel=np.zeros_like(mj_sim.data.qpos), act=np.zeros_like(mj_sim.data.ctrl), udd_state={}
    )

    for u in us:
        mj_sim.data.ctrl[0] = u
        mj_sim.forward()
        mj_sim.step()
        mj_viewer.render()
        sleep(0.01)

    import glfw
    glfw.terminate()

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    matplotlib.use("TkAgg")
    import numpy as np
    from matplotlib import rcParams

    # Improve overall aesthetics
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['DejaVu Sans']
    rcParams['font.size'] = 12
    rcParams['axes.titlepad'] = 20

    # Simulation parameters
    freq = 8
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim((0, 2 * np.pi * freq))
    ax.set_ylim((-arm1_scale_lf*2 * 1.1, arm1_scale_lf*2 * 1.1))

    # Customize plot background and grid
    ax.set_facecolor('#c6c6c6')  # Set the background color to grey
    ax.set_title('Control Input Trajectory', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Control Input', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')  # Set grid lines to white

    line, = ax.plot([], [], lw=2, linestyle='-', color='royalblue', marker='', label='Control')
    head, = ax.plot([], [], 'ro', markersize=8)  # Red dot at the front of the trajectory

    # Add a legend
    ax.legend(loc='upper right')


    # Initialize the animation
    def init():
        line.set_data([], [])
        head.set_data([], [])
        return (line, head)


    # Define the animation update
    def animate(i):
        x = time[:i]
        y = us[:i]
        line.set_data(x, y)
        if i > 0:  # Update the head's position if there are points to plot
            head.set_data(x[-1], y[-1])
        return (line, head)


    # Create the animation
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=timesteps, interval=10, blit=True)
    # ani.save('reacher_lf2.mp4', fps=100, extra_args=['-vcodec', 'libx264'])
    plt.show()  # Prevent the static plot from showing in Jupyter notebooks or Python script

"""Speed tests for the CBF solver

We evaluate the speed of the solver NOT just via the QP solve but via the whole process
(solving for the nominal control input, constructing the QP matrices, and then solving). 
This provides a more accurate view of what the controller frequency would actually be if
deployed on the robot.

These test cases can also be used to check that modifications to the CBF implementation 
do not significantly degrade performance
"""

import unittest
from typing import Callable
import time
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt

from cbfpy import CBF, CLFCBF

from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from oscbf.core.controllers import PoseTaskTorqueController, PoseTaskVelocityController
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.utils.general_utils import find_assets_dir

# Seed RNG for repeatability
np.random.seed(0)


# TODO Make this work if there are additional arguments for the barrier function
def eval_speed(
    controller_func: Callable,
    states: np.ndarray,
    des_states: np.ndarray,
    verbose: bool = True,
    plot: bool = True,
) -> float:
    """Tests the speed of a controller function via evaluation on a set of inputs

    Timing details (average solve time / Hz, distributions of times, etc.) can be printed to the terminal
    or visualized in plots, via the `verbose` and `plot` inputs

    Args:
        controller_func (Callable): Function to time. This should be the highest-level CBF-based controller
            function which includes the nominal controller, QP construction, and QP solve
        states (np.ndarray): Set of states to evaluate on, shape (num_evals, state_dim)
        des_states (np.ndarray): Set of desired states to evaluate on, shape (num_evals, des_state_dim)
        verbose (bool, optional): Whether to print timing details to the terminal. Defaults to True.
        plot (bool, optional): Whether to visualize the distribution of solve times. Defaults to True.

    Returns:
        float: Average solver Hz
    """
    assert isinstance(controller_func, Callable)
    assert isinstance(states, np.ndarray)
    assert isinstance(des_states, np.ndarray)
    assert isinstance(verbose, bool)
    assert isinstance(plot, bool)
    assert states.shape[0] > 1
    assert states.shape[0] == des_states.shape[0]
    controller_func: Callable = jax.jit(controller_func)

    # Do an initial solve to jit-compile the function
    start_time = time.perf_counter()
    u = controller_func(states[0], des_states[0])
    first_solve_time = time.perf_counter() - start_time

    # Solve for the remainder of the controls using the jit-compiled controller
    times = []
    for i in range(1, states.shape[0]):
        start_time = time.perf_counter()
        u = controller_func(states[i], des_states[i]).block_until_ready()
        times.append(time.perf_counter() - start_time)
    times = np.asarray(times)
    avg_solve_time = np.mean(times)
    max_solve_time = np.max(times)
    avg_hz = 1 / avg_solve_time
    worst_case_hz = 1 / max_solve_time

    if verbose:
        # Print info about solver stats
        print(f"Solved for the first control input in {first_solve_time} seconds")
        print(f"Average solve time: {avg_solve_time} seconds")
        print(f"Average Hz: {avg_hz}")
        print(f"Worst-case solve time: {max_solve_time}")
        print(f"Worst-case Hz: {worst_case_hz}")
        print(
            "NOTE: Worst-case behavior might be inaccurate due to how the OS manages background processes"
        )

    if plot:
        # Create a plot to visualize the distribution of times
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].hist(times, 20)
        axs[0, 0].set_title("Solve Times")
        axs[0, 0].set_ylabel("Frequency")
        axs[0, 0].set_xscale("log")
        axs[0, 1].boxplot(times, vert=False)
        axs[0, 1].set_title("Solve Times")
        axs[0, 1].set_xscale("log")
        axs[1, 0].hist(1 / times, 20)
        axs[1, 0].set_title("Hz")
        axs[1, 0].set_ylabel("Frequency")
        axs[1, 1].boxplot(1 / times, vert=False)
        axs[1, 1].set_title("Hz")
        plt.show()

    return avg_hz


@jax.tree_util.register_static
class EESafeSetTorqueConfig(OSCBFTorqueConfig):

    def __init__(
        self,
        robot: Manipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
    ):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        h_ee_safe_set = jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])
        return h_ee_safe_set

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


# TODO clean this up alongside the velocity control test
@jax.tree_util.register_static
class OSCBFTorqueControlTest:
    """Test the speed of the manipulator demo, using randomly sampled states"""

    def __init__(self):
        self.robot = load_panda()
        ee_pos_min = np.array([0.15, -0.25, 0.25])
        ee_pos_max = np.array([0.75, 0.25, 0.75])
        self.config = EESafeSetTorqueConfig(self.robot, ee_pos_min, ee_pos_max)
        self.cbf = CBF.from_config(self.config)

        self.pos_min = self.config.pos_min
        self.pos_max = self.config.pos_max

        kp_pos = 50.0
        kp_rot = 20.0
        kd_pos = 20.0
        kd_rot = 10.0
        kp_joint = 10.0
        kd_joint = 5.0
        self.osc_controller = PoseTaskTorqueController(
            n_joints=self.robot.num_joints,
            kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
            kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
            kp_joint=kp_joint,
            kd_joint=kd_joint,
            # Note: torque limits will be enforced via the QP. We'll set them to None here
            # because we don't want to clip the values before the QP
            tau_min=None,
            tau_max=None,
        )

    def sample_joint_states(self, num_samples: int) -> np.ndarray:
        num_joints = self.robot.num_joints
        default_joint_pos = np.array(
            [0, -np.pi / 3, 0, -5 * np.pi / 6, 0, np.pi / 2, 0]
        )
        joint_angles = np.random.multivariate_normal(
            default_joint_pos, np.diag(0.1 * np.ones(num_joints)), num_samples
        )
        joint_velocities = np.random.multivariate_normal(
            np.zeros(num_joints), np.diag(0.1 * np.ones(num_joints)), num_samples
        )
        return np.column_stack([joint_angles, joint_velocities])

    def sample_ee_states(self, num_samples: int) -> np.ndarray:
        # Sample positions uniformly inside the keep-in region
        positions = np.asarray(self.pos_min) + np.random.rand(
            num_samples, 3
        ) * np.subtract(self.pos_max, self.pos_min)
        # Assume identity rotation for now
        # NOTE: switched to flat rotation matrix instead of quaternion
        rmats_flat = np.tile(np.eye(3).ravel(), (num_samples, 1))
        # Sample velocities uniformly in [-1, 1]
        velocities = -1.0 + 2 * np.random.rand(num_samples, 3)
        # Assume no angular velocity for now
        omegas = np.zeros((num_samples, 3))
        return np.column_stack([positions, rmats_flat, velocities, omegas])

    @jax.jit
    def controller(
        self,
        z: ArrayLike,
        z_ee_des: ArrayLike,
    ):
        q = z[: self.robot.num_joints]
        qdot = z[self.robot.num_joints :]
        M, M_inv, g, c, J, ee_tmat = self.robot.torque_control_matrices(q, qdot)
        # Set nullspace desired joint position
        nullspace_posture_goal = jnp.array(
            [
                0.0,
                -jnp.pi / 6,
                0.0,
                -3 * jnp.pi / 4,
                0.0,
                5 * jnp.pi / 9,
                0.0,
            ]
        )

        # Compute nominal control
        u_nom = self.osc_controller(
            q,
            qdot,
            pos=ee_tmat[:3, 3],
            rot=ee_tmat[:3, :3],
            des_pos=z_ee_des[:3],
            des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
            des_vel=z_ee_des[12:15],
            des_omega=z_ee_des[15:18],
            des_accel=jnp.zeros(3),
            des_alpha=jnp.zeros(3),
            des_q=nullspace_posture_goal,
            des_qdot=jnp.zeros(self.robot.num_joints),
            J=J,
            M=M,
            M_inv=M_inv,
            g=g,
            c=c,
        )
        # Apply the CBF safety filter
        return self.cbf.safety_filter(z, u_nom)

    def test_speed(self, verbose: bool = True, plot: bool = True):
        n_samples = 10000
        states = self.sample_joint_states(n_samples)
        des_ee_states = self.sample_ee_states(n_samples)
        avg_hz = eval_speed(self.controller, states, des_ee_states, verbose, plot)
        return avg_hz


class SpeedTest(unittest.TestCase):
    """Test case to guarantee that the CBFs run at least at kilohertz rates"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manip_torque_test = OSCBFTorqueControlTest()

    def test_manip_torque_control(self):
        avg_hz = self.manip_torque_test.test_speed(verbose=False, plot=False)
        print("Manipulator (torque control) average Hz: ", avg_hz)
        self.assertTrue(avg_hz >= 1000)


if __name__ == "__main__":
    unittest.main()
"""Testing self-collision avoidance"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from cbfpy import CBF

from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.manipulation_env import FrankaTorqueControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.controllers import PoseTaskTorqueController

@jax.tree_util.register_static
class SelfCollisionConfig(OSCBFTorqueConfig):

    def __init__(
        self,
        robot: Manipulator,
    ):
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-3
        super().__init__(robot, rot_obj_weight=0.1)

    def h_2(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]

        # Self collision avoidance
        robot_collision_pos_rad = self.robot.link_self_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3]
        pairs = jnp.asarray(self.robot.self_collision_pairs)
        pos_a = robot_collision_positions[pairs[:, 0]]
        pos_b = robot_collision_positions[pairs[:, 1]]
        rad_a = robot_collision_radii[pairs[:, 0]]
        rad_b = robot_collision_radii[pairs[:, 1]]
        center_deltas = pos_b - pos_a
        h_self_collision = jnp.linalg.norm(center_deltas, axis=-1) - rad_a - rad_b

        # Base self collision avoidance
        base_position = jnp.asarray(self.robot.base_self_collision_position)
        base_radius = self.robot.base_self_collision_radius
        base_sc_idxs = jnp.asarray(self.robot.base_self_collision_idxs)
        base_sc_deltas = robot_collision_positions[base_sc_idxs] - base_position
        h_base_self_collision = (
            jnp.linalg.norm(base_sc_deltas, axis=-1)
            - robot_collision_radii[base_sc_idxs]
            - base_radius
        )

        # Joint Limit Avoidance
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        return jnp.concatenate([h_self_collision, h_base_self_collision, h_joint_limits, h_singularity])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
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
    u_nom = osc_controller(
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
        des_qdot=jnp.zeros(robot.num_joints),
        J=J,
        M=M,
        M_inv=M_inv,
        g=g,
        c=c,
    )
    # Apply the CBF safety filter
    return cbf.safety_filter(z, u_nom)


def main():
    robot = load_panda()
    config = SelfCollisionConfig(robot)
    cbf = CBF.from_config(config)
    env = FrankaTorqueControlEnv(
        load_floor=False,
        bg_color=(1, 1, 1),
        real_time=True,
    )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraPitch=-27.80,
        cameraYaw=36.80,
        cameraTargetPosition=(0.08, 0.49, -0.04),
    )

    kp_pos = 50.0
    kp_rot = 20.0
    kd_pos = 20.0
    kd_rot = 10.0
    kp_joint = 10.0
    kd_joint = 5.0
    osc_controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
        kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
        kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
        kp_joint=kp_joint,
        kd_joint=kd_joint,
        # Note: torque limits will be enforced via the QP. We'll set them to None here
        # because we don't want to clip the values before the QP
        tau_min=None,
        tau_max=None,
    )

    @jax.jit
    def compute_control_jit(z, z_des):
        return compute_control(robot, osc_controller, cbf, z, z_des)

    while True:
        joint_state = env.get_joint_state()
        ee_state_des = env.get_desired_ee_state()
        tau = compute_control_jit(joint_state, ee_state_des)
        env.apply_control(tau)
        env.step()


if __name__ == "__main__":
    main()

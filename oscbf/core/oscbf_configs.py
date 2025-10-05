"""OSCBF Configs

Includes the dynamics and task-consistent objective functions for both
velocity control and torque control.

(These objectives assume that the desired task is tracking a desired pose
with a secondary null space posture target)

Note: The CBFs themselves are not included in this file. These should
inherit from these configs, and implement the h_1/h_2 methods
depending on the desired safety criteria. For examples of this,
see the `oscbf/examples` folder
"""

import jax.numpy as jnp
import numpy as np
from cbfpy import CBFConfig

from oscbf.core.manipulator import Manipulator


class OSCBFTorqueConfig(CBFConfig):
    """CBF Configuration for safe torque-controlled manipulation

    State: z = [q, qdot] (length = 2 * num_joints)
    Control: u = [joint torques] (length = num_joints)

    Args:
        robot (Manipulator): The robot model (kinematics and dynamics)
        pos_obj_weight (float, optional): Objective function weight for the safety filter's
            impact on the end-effector's position tracking performance. Defaults to 1.0.
        rot_obj_weight (float, optional): Objective function weight for the safety filter's
            impact on the end-effector's orientation tracking performance. Defaults to 1.0.
        joint_obj_weight (float, optional): Objective function weight for the safety filter's
            impact on the joint space tracking performance. Defaults to 1.0.
        compensate_centrifugal_coriolis (bool, optional): Whether to compensate for centrifugal/coriolis
            terms in the robot dynamics. This term is often neglected (set to False). Defaults to True.
        init_args (tuple, optional): Optional initial seed for testing additional CBFpy barrier arguments.
            Defaults to ().
    """

    def __init__(
        self,
        robot: Manipulator,
        pos_obj_weight: float = 1.0,
        rot_obj_weight: float = 1.0,
        joint_obj_weight: float = 1.0,
        compensate_centrifugal_coriolis: bool = True,
        init_args: tuple = (),
    ):
        assert isinstance(robot, Manipulator)
        assert isinstance(pos_obj_weight, (tuple, float)) and pos_obj_weight >= 0
        assert isinstance(rot_obj_weight, (tuple, float)) and rot_obj_weight >= 0
        assert isinstance(joint_obj_weight, (tuple, float)) and joint_obj_weight >= 0
        assert isinstance(compensate_centrifugal_coriolis, bool)
        self.robot = robot
        self.num_joints = self.robot.num_joints
        self.task_dim = 6  # Pose
        self.is_redundant = self.robot.num_joints > self.task_dim
        self.compensate_centrifugal_coriolis = compensate_centrifugal_coriolis
        self.pos_obj_weight = float(pos_obj_weight)
        self.rot_obj_weight = float(rot_obj_weight)
        self.joint_space_obj_weight = float(joint_obj_weight)
        # Store the diagonal of W.T @ W for the task and joint space weighting matrices
        # This assumes that W is a diagonal matrix with only positive values
        self.W_T_W_task_diag = tuple(
            np.array([self.pos_obj_weight] * 3 + [self.rot_obj_weight] * 3) ** 2
        )
        self.W_T_W_joint_diag = tuple(
            np.array([self.joint_space_obj_weight] * self.num_joints) ** 2
        )

        super().__init__(
            n=self.num_joints * 2,
            m=self.num_joints,
            u_min=-np.asarray(robot.joint_max_forces),
            u_max=np.asarray(robot.joint_max_forces),
            init_args=init_args,
        )

    def f(self, z, *args, **kwargs):
        q = z[: self.num_joints]
        q_dot = z[self.num_joints : self.num_joints * 2]
        M = self.robot.mass_matrix(q)
        M_inv = jnp.linalg.inv(M)
        bias = self.robot.gravity_vector(q)
        if self.compensate_centrifugal_coriolis:
            bias += self.robot.centrifugal_coriolis_vector(q, q_dot)
        return jnp.concatenate(
            [
                q_dot,  # Joint velocity
                -M_inv @ bias,  # Joint acceleration
            ]
        )

    def g(self, z, *args, **kwargs):
        q = z[: self.num_joints]
        M = self.robot.mass_matrix(q)
        M_inv = jnp.linalg.inv(M)
        return jnp.block(
            [
                [jnp.zeros((self.num_joints, self.m))],
                [M_inv],
            ]
        )

    def _P(self, z, *args, **kwargs):
        """Helper function to compute the QP P matrix. This get reused in both the P and q QP functions"""
        q = z[: self.num_joints]
        transforms = self.robot.joint_to_world_transforms(q)
        M = self.robot._mass_matrix(transforms)
        M_inv = jnp.linalg.inv(M)
        J = self.robot._ee_jacobian(transforms)
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia
        NT = jnp.eye(self.num_joints) - J.T @ J_bar.T

        W_T_W_joint = jnp.diag(jnp.asarray(self.W_T_W_joint_diag))
        W_T_W_task = jnp.diag(jnp.asarray(self.W_T_W_task_diag))

        return (
            NT.T @ M_inv.T @ W_T_W_joint @ M_inv @ NT
            + M_inv.T @ J.T @ W_T_W_task @ J @ M_inv
        )

    def P(self, z, u_des, *args, **kwargs):
        return self._P(z)

    def q(self, z, u_des, *args, **kwargs):
        return -u_des.T @ self._P(z)


class OSCBFVelocityConfig(CBFConfig):
    """CBF Configuration for safe velocity-controlled manipulation

    State: z = [q] (length = num_joints)
    Control: u = [joint velocities] (length = num_joints)

    Args:
        robot (Manipulator): The robot model (kinematics and dynamics)
        pos_obj_weight (float, optional): Objective function weight for the safety filter's
            impact on the end-effector's position tracking performance. Defaults to 1.0.
        rot_obj_weight (float, optional): Objective function weight for the safety filter's
            impact on the end-effector's orientation tracking performance. Defaults to 1.0.
        joint_obj_weight (float, optional): Objective function weight for the safety filter's
            impact on the joint space tracking performance. Defaults to 1.0.
        init_args (tuple, optional): Optional initial seed for testing additional CBFpy barrier arguments.
            Defaults to ().
    """

    def __init__(
        self,
        robot: Manipulator,
        pos_obj_weight: float = 1.0,
        rot_obj_weight: float = 1.0,
        joint_obj_weight: float = 1.0,
        init_args: tuple = (),
    ):
        assert isinstance(robot, Manipulator)
        assert isinstance(pos_obj_weight, (tuple, float)) and pos_obj_weight >= 0
        assert isinstance(rot_obj_weight, (tuple, float)) and rot_obj_weight >= 0
        assert isinstance(joint_obj_weight, (tuple, float)) and joint_obj_weight >= 0
        self.robot = robot
        self.num_joints = self.robot.num_joints
        self.task_dim = 6  # Pose
        self.is_redundant = self.robot.num_joints > self.task_dim

        self.pos_obj_weight = float(pos_obj_weight)
        self.rot_obj_weight = float(rot_obj_weight)
        self.joint_space_obj_weight = float(joint_obj_weight)
        # Store the diagonal of W.T @ W for the task and joint space weighting matrices
        # This assumes that W is a diagonal matrix with only positive values
        self.W_T_W_task_diag = tuple(
            np.array([self.pos_obj_weight] * 3 + [self.rot_obj_weight] * 3) ** 2
        )
        self.W_T_W_joint_diag = tuple(
            np.array([self.joint_space_obj_weight] * self.num_joints) ** 2
        )

        super().__init__(
            n=self.num_joints,
            m=self.num_joints,
            u_min=-np.asarray(robot.joint_max_velocities),
            u_max=np.asarray(robot.joint_max_velocities),
            init_args=init_args,
        )

    def f(self, z, *args, **kwargs):
        return jnp.zeros(self.n)

    def g(self, z, *args, **kwargs):
        return jnp.eye(self.num_joints)

    def _P(self, z, *args, **kwargs):
        q = z
        transforms = self.robot.joint_to_world_transforms(q)
        J = self.robot._ee_jacobian(transforms)
        M = self.robot._mass_matrix(transforms)
        M_inv = jnp.linalg.inv(M)
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia
        J_hash = J_bar
        N = jnp.eye(self.num_joints) - J_hash @ J
        W_T_W_joint = jnp.diag(jnp.asarray(self.W_T_W_joint_diag))
        W_T_W_task = jnp.diag(jnp.asarray(self.W_T_W_task_diag))

        return N.T @ W_T_W_joint @ N + J.T @ W_T_W_task @ J

    def P(self, z, u_des, *args, **kwargs):
        return self._P(z)

    def q(self, z, u_des, *args, **kwargs):
        return -u_des.T @ self._P(z)

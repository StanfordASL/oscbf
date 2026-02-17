"""Simulation environment for manipulator end-effector pose tracking"""

import time
from typing import Optional, Dict, Tuple
from functools import partial
import argparse

import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.utils.visualization import visualize_3D_box
from oscbf.utils.general_utils import stdout_redirected
from oscbf.assets import ASSETS_DIR
from oscbf.core.controllers import PoseTaskVelocityController, PoseTaskTorqueController
from oscbf.utils.trajectory import TaskTrajectory


class ManipulationEnv:
    """Simulation environment for manipulator end-effector pose tracking

    Args:
        urdf (str): Path to the URDF file of the robot
        control_mode (str): Control mode, either "torque" or "velocity"
        xyz_min (Optional[ArrayLike]): Minimum bounds of the safe region, shape (3,). Defaults to None.
        xyz_max (Optional[ArrayLike]): Maximum bounds of the safe region, shape (3,). Defaults to None.
        target_pos (ArrayLike): Initial position of the target, shape (3,). Defaults to (0.5, 0, 0.5).
        q_init (Optional[ArrayLike]): Initial joint positions of the robot, shape (num_joints,). Defaults to None.
        traj (Optional[TaskTrajectory]): Task-space trajectory for the target to follow. Defaults to None, in
            which case the target's position and orientation can be controlled by the user in the GUI.
        collision_data (Optional[Dict[str, Tuple[ArrayLike]]]): Collision information, represented as spheres.
            The dictionary should have keys "positions" and "radii". Defaults to None.
        wb_xyz_min (Optional[ArrayLike]): Minimum bounds of the whole-body safe region, shape (3,). Defaults to None.
        wb_xyz_max (Optional[ArrayLike]): Maximum bounds of the whole-body safe region, shape (3,). Defaults to None.
        bg_color (Optional[ArrayLike]): RGB background color of the simulation. Defaults to None
            (use default background color)
        load_floor (bool): Whether to load a floor into the simulation. Defaults to True.
        qdot_max (Optional[ArrayLike]): Maximum joint velocities, shape (num_joints,). Defaults to None.
        tau_max (Optional[ArrayLike]): Maximum joint torques, shape (num_joints,). Defaults to None.
        real_time (bool): Whether to run the simulation in "real time". Defaults to False.
        timestep (float): Simulation timestep. Defaults to 1/240 (Same as PyBullet's default timestep).
        load_table (bool): Whether to load a table into the simulation. Defaults to False.
    """

    def __init__(
        self,
        urdf: str,
        control_mode: str,
        xyz_min: Optional[ArrayLike] = None,
        xyz_max: Optional[ArrayLike] = None,
        target_pos: ArrayLike = (0.5, 0, 0.5),
        q_init: Optional[ArrayLike] = None,
        traj: Optional[TaskTrajectory] = None,
        collision_data: Optional[Dict[str, Tuple[ArrayLike]]] = None,
        wb_xyz_min: Optional[ArrayLike] = None,
        wb_xyz_max: Optional[ArrayLike] = None,
        bg_color: Optional[ArrayLike] = None,
        load_floor: bool = True,
        qdot_max: Optional[ArrayLike] = None,
        tau_max: Optional[ArrayLike] = None,
        real_time: bool = False,
        timestep: float = 1 / 240,
        load_table: bool = False,
    ):
        assert isinstance(urdf, str)
        self.urdf = urdf
        assert control_mode in ["torque", "velocity"]
        self.control_mode = control_mode
        assert isinstance(traj, TaskTrajectory) or traj is None
        self.traj = traj
        with stdout_redirected():
            self.client: pybullet = BulletClient(pybullet.GUI)
        assert isinstance(timestep, float) and timestep > 0
        self.client.setTimeStep(timestep)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = self.client.loadURDF(
            urdf,
            useFixedBase=True,
            flags=self.client.URDF_USE_INERTIA_FROM_FILE
            | self.client.URDF_MERGE_FIXED_LINKS,
        )
        assert isinstance(load_table, bool)
        table_z_offset = -0.35
        if load_table:
            self.table = pybullet.loadURDF(
                "table/table.urdf",
                [0.5, 0, table_z_offset],
                globalScaling=0.7,
                baseOrientation=pybullet.getQuaternionFromEuler((0, 0, np.pi / 2)),
            )
        if load_floor:
            if load_table:
                self.floor = self.client.loadURDF("plane.urdf", [0, 0, table_z_offset])
            else:
                self.floor = self.client.loadURDF("plane.urdf")
        self.client.configureDebugVisualizer(self.client.COV_ENABLE_GUI, 0)
        if bg_color is not None:
            assert len(bg_color) == 3
            self.client.configureDebugVisualizer(rgbBackground=bg_color)
        self.num_joints = self.client.getNumJoints(self.robot)
        # "Unlock" the joints
        self.client.setJointMotorControlArray(
            self.robot,
            list(range(self.num_joints)),
            pybullet.VELOCITY_CONTROL,
            forces=[0.1] * self.num_joints,
        )
        self.target = self.client.loadURDF(
            str(ASSETS_DIR / "point_robot.urdf"),
            basePosition=target_pos,
            # baseOrientation=self.client.getQuaternionFromEuler([np.pi, 0, 0]),
            globalScaling=0.2,
        )
        self.target_mass = 1.0
        self.client.changeVisualShape(self.target, -1, rgbaColor=[1, 0, 0, 0.6])
        self.client.changeDynamics(self.target, -1, linearDamping=10, angularDamping=30)

        # If this environment is set up for safe-set-invariance, visualize the safe box
        if xyz_min is not None and xyz_max is not None:
            self.xyz_min = np.asarray(xyz_min)
            self.xyz_max = np.asarray(xyz_max)
            assert self.xyz_min.shape == self.xyz_max.shape == (3,)
            self.box_id = visualize_3D_box(
                [self.xyz_min, self.xyz_max], rgba=(0, 1, 0, 0.3)
            )
        else:
            self.box_id = None
            self.xyz_min = None
            self.xyz_max = None

        # Slight HACK: Duplicate logic for whole-body safe set
        if wb_xyz_min is not None and wb_xyz_max is not None:
            self.wb_xyz_min = np.asarray(wb_xyz_min)
            self.wb_xyz_max = np.asarray(wb_xyz_max)
            assert self.wb_xyz_min.shape == self.wb_xyz_max.shape == (3,)
            self.wb_box_id = visualize_3D_box(
                [self.wb_xyz_min, self.wb_xyz_max], rgba=(0, 1, 0, 0.3)
            )
        else:
            self.wb_box_id = None
            self.wb_xyz_min = None
            self.wb_xyz_max = None

        self.qdot_max = qdot_max
        self.tau_max = tau_max

        self.real_time = real_time
        self.last_time = time.time()

        # Disable collisions for all links and base of the robot with the target
        for i in range(-1, self.num_joints + 1):
            self.client.setCollisionFilterPair(self.robot, self.target, i, -1, 0)
        disable_robot_floor_collisions = False
        disable_target_floor_collisions = True
        if load_floor:
            if disable_robot_floor_collisions:
                for i in range(-1, self.num_joints + 1):
                    self.client.setCollisionFilterPair(self.robot, self.floor, i, -1, 0)
            if disable_target_floor_collisions:
                self.client.setCollisionFilterPair(self.floor, self.target, -1, -1, 0)

        # Disable collisions between the target and the table
        disable_robot_table_collisions = False
        disable_target_table_collisions = True
        if load_table:
            if disable_target_table_collisions:
                self.client.setCollisionFilterPair(self.table, self.target, -1, -1, 0)
            if disable_robot_table_collisions:
                for i in range(-1, self.num_joints + 1):
                    self.client.setCollisionFilterPair(self.robot, self.table, i, -1, 0)
            # Add a stand for the robot
            self.robot_stand_id = visualize_3D_box(
                np.asarray([(-0.2, -0.1, -0.35), (0.1, 0.1, 0)]), rgba=(1, 1, 1, 1)
            )

        # Set initial joint positions if provided
        if q_init is not None:
            assert len(q_init) == self.num_joints
            for joint_index in range(len(q_init)):
                self.client.resetJointState(
                    self.robot, joint_index, q_init[joint_index]
                )

        # Handle collision info
        if collision_data is not None:
            self.collision_positions = collision_data["positions"]
            self.collision_radii = collision_data["radii"]
            self.collision_ids = []
            assert len(self.collision_positions) == len(self.collision_radii)
            for p, r in zip(self.collision_positions, self.collision_radii):
                # coll_id = self.client.createCollisionShape(
                #     self.client.GEOM_SPHERE,
                #     radius=r,
                #     collisionFramePosition=p,
                # )
                # We don't actually want to create a collision shape,
                # because then it would be hard to tell if we are avoiding it
                coll_id = -1
                vis_id = self.client.createVisualShape(
                    self.client.GEOM_SPHERE,
                    radius=r,
                    visualFramePosition=p,
                    rgbaColor=[0, 0, 1, 0.5],
                )
                body_id = self.client.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=coll_id,
                    baseVisualShapeIndex=vis_id,
                )
                self.collision_ids.append(body_id)
        else:
            self.collision_positions = None
            self.collision_radii = None
            self.collision_ids = None

        self.client.setGravity(0, 0, -9.81)
        self.dt = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        self.t = 0

    def get_joint_state(self) -> Array:
        joint_angles = []
        joint_velocities = []
        joint_states = self.client.getJointStates(
            self.robot, list(range(self.num_joints))
        )
        joint_angles = [joint_state[0] for joint_state in joint_states]
        joint_velocities = [joint_state[1] for joint_state in joint_states]
        return np.array([*joint_angles, *joint_velocities])

    def get_desired_ee_state(self) -> Array:
        # Follow a desired task-space trajectory if provided
        if self.traj is not None:
            pos = self.traj.position(self.t)
            rot = self.traj.rotation(self.t).ravel()
            vel = self.traj.velocity(self.t)
            omega = self.traj.omega(self.t)
            # Update the target's visuals to match the desired state
            # HACK: Assume fixed rotation (TODO get a conversion function in here)
            # quat = rotation_to_xyzw(rot.reshape(3, 3))
            quat = np.array([0, 0, 0, 1])
            self.client.resetBasePositionAndOrientation(self.target, pos, quat)
            self.client.resetBaseVelocity(self.target, vel, omega)
            return np.array([*pos, *rot, *vel, *omega])
        # Otherwise, respond to GUI inputs from the user
        pos, orn = self.client.getBasePositionAndOrientation(self.target)
        vel, omega = self.client.getBaseVelocity(self.target)

        # HACK: reset the angular vel of the target to 0
        # Sometimes, the target can start spinning out of control -- this fixes that
        self.client.resetBaseVelocity(self.target, vel, [0, 0, 0])

        # rot = np.ravel(self.client.getMatrixFromQuaternion(orn))
        # Rotate the target so that the Franka EE naturaly faces downwards instead of upwards
        # Also, flatten the rotation matrix to a 1D array
        rot = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        ).ravel()
        # TEMP Ignore angular velocity for now
        omega = np.zeros(3)
        return np.array([*pos, *rot, *vel, *omega])

    def apply_control(self, u: Array) -> None:
        if self.control_mode == "velocity":
            if self.qdot_max is not None:
                u = np.clip(u, -self.qdot_max, self.qdot_max)
            if self.tau_max is not None:
                self.client.setJointMotorControlArray(
                    self.robot,
                    list(range(self.num_joints)),
                    self.client.VELOCITY_CONTROL,
                    targetVelocities=u,
                    forces=self.tau_max,
                )
            else:
                self.client.setJointMotorControlArray(
                    self.robot,
                    list(range(self.num_joints)),
                    self.client.VELOCITY_CONTROL,
                    targetVelocities=u,
                )
        else:  # Torque control
            if self.tau_max is not None:
                u = np.clip(u, -self.tau_max, self.tau_max)
            self.client.setJointMotorControlArray(
                self.robot,
                list(range(self.num_joints)),
                self.client.TORQUE_CONTROL,
                forces=u,
            )
        # Gravity compensation for the target robot so it doesn't fly away
        self.client.applyExternalForce(
            self.target,
            -1,
            [0, 0, 9.81 * self.target_mass],
            self.client.getBasePositionAndOrientation(self.target)[0],
            self.client.WORLD_FRAME,
        )

    def step(self):
        self.client.stepSimulation()
        self.t += self.dt
        if self.real_time:
            time.sleep(max(0, self.dt - (time.time() - self.last_time)))
            self.last_time = time.time()


class FrankaTorqueControlEnv(ManipulationEnv):
    """Simulation environment for Franka end-effector pose tracking, with torque control"""

    def __init__(
        self,
        xyz_min=None,
        xyz_max=None,
        target_pos=(0.5, 0, 0.5),
        q_init=(0, -np.pi / 3, 0, -5 * np.pi / 6, 0, np.pi / 2, 0),
        traj=None,
        collision_data=None,
        wb_xyz_min=None,
        wb_xyz_max=None,
        bg_color=None,
        load_floor=True,
        real_time=False,
        timestep=1 / 240,
        load_table=False,
    ):
        super().__init__(
            str(ASSETS_DIR / "franka_panda/panda.urdf"),
            "torque",
            xyz_min,
            xyz_max,
            target_pos,
            q_init,
            traj,
            collision_data,
            wb_xyz_min,
            wb_xyz_max,
            bg_color,
            load_floor,
            qdot_max=np.array((2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)),
            tau_max=np.array((87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)),
            real_time=real_time,
            timestep=timestep,
            load_table=load_table,
        )


class FrankaVelocityControlEnv(ManipulationEnv):
    """Simulation environment for Franka end-effector pose tracking, with velocity control"""

    def __init__(
        self,
        xyz_min=None,
        xyz_max=None,
        target_pos=(0.5, 0, 0.5),
        q_init=(0, -np.pi / 3, 0, -5 * np.pi / 6, 0, np.pi / 2, 0),
        traj=None,
        collision_data=None,
        wb_xyz_min=None,
        wb_xyz_max=None,
        bg_color=None,
        load_floor=True,
        real_time=False,
        timestep=1 / 240,
        load_table=False,
    ):
        super().__init__(
            str(ASSETS_DIR / "franka_panda/panda.urdf"),
            "velocity",
            xyz_min,
            xyz_max,
            target_pos,
            q_init,
            traj,
            collision_data,
            wb_xyz_min,
            wb_xyz_max,
            bg_color,
            load_floor,
            qdot_max=np.array((2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)),
            tau_max=np.array((87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)),
            real_time=real_time,
            timestep=timestep,
            load_table=load_table,
        )


# TODO make this more general -- the des_q only works for a 7DOF robot
# (same thing applies for the nominal torque controller)
@partial(jax.jit, static_argnums=(0, 1))
def nominal_velocity_controller(
    manipulator: Manipulator, controller: PoseTaskVelocityController, q, z_ee_des
):
    M_inv, J, ee_tmat = manipulator.dynamically_consistent_velocity_control_matrices(q)
    pos = ee_tmat[:3, 3]
    rot = ee_tmat[:3, :3]
    des_pos = z_ee_des[:3]
    des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
    des_vel = z_ee_des[12:15]
    des_omega = z_ee_des[15:18]
    # Set nullspace desired joint position
    des_q = jnp.array(
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
    return controller(
        q, pos, rot, des_pos, des_rot, des_vel, des_omega, des_q, J, M_inv
    )


@partial(jax.jit, static_argnums=(0, 1))
def nominal_torque_controller(
    manipulator: Manipulator, controller: PoseTaskTorqueController, z, z_ee_des
):
    q = z[: manipulator.num_joints]
    qdot = z[manipulator.num_joints :]
    M, M_inv, g, c, J, ee_tmat = manipulator.torque_control_matrices(q, qdot)
    pos = ee_tmat[:3, 3]
    rot = ee_tmat[:3, :3]
    des_pos = z_ee_des[:3]
    des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
    des_vel = z_ee_des[12:15]
    des_omega = z_ee_des[15:18]
    des_accel = jnp.zeros(3)
    des_alpha = jnp.zeros(3)
    # Set nullspace desired joint position
    des_q = jnp.array(
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
    des_qdot = jnp.zeros(manipulator.num_joints)
    return controller(
        q,
        qdot,
        pos,
        rot,
        des_pos,
        des_rot,
        des_vel,
        des_omega,
        des_accel,
        des_alpha,
        des_q,
        des_qdot,
        J,
        M,
        M_inv,
        g,
        c,
    )


def test_velocity_control():
    env = FrankaVelocityControlEnv()
    robot = load_panda()
    kp_pos = 5.0
    kp_rot = 2.0
    kp_task = np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)])
    kp_joint = 1.0
    controller = PoseTaskVelocityController(
        robot.num_joints,
        kp_task,
        kp_joint,
        -1 * np.asarray(robot.joint_max_velocities),
        robot.joint_max_velocities,
    )

    while True:
        z = env.get_joint_state()
        z_des = env.get_desired_ee_state()
        q = z[:7]
        u = nominal_velocity_controller(robot, controller, q, z_des)
        env.apply_control(u)
        env.step()


def test_torque_control():
    env = FrankaTorqueControlEnv()
    robot = load_panda()
    kp_pos = 50.0
    kp_rot = 20.0
    kd_pos = 20.0
    kd_rot = 10.0
    controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
        kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
        kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
        kp_joint=10.0,
        kd_joint=5.0,
        tau_min=None,
        tau_max=None,
    )

    while True:
        z = env.get_joint_state()
        z_ee_des = env.get_desired_ee_state()
        u = nominal_torque_controller(robot, controller, z, z_ee_des)
        env.apply_control(u)
        env.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test velocity or torque control on the manipulator."
    )
    parser.add_argument(
        "--control",
        choices=["velocity", "torque"],
        default="velocity",
        help="Specify the control mode to test: 'velocity' or 'torque'.",
    )
    args = parser.parse_args()

    if args.control == "velocity":
        print("Testing velocity control")
        test_velocity_control()
    elif args.control == "torque":
        print("Testing torque control")
        test_torque_control()

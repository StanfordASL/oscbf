"""Test cases for the Manipulator class"""

import unittest

from oscbf.core.manipulator import Manipulator
from oscbf.assets import ASSETS_DIR

import pybullet
import jax.numpy as jnp
import numpy as np

try:
    import pinocchio as pin

    PIN_INSTALLED = True
except ImportError:
    PIN_INSTALLED = False

URDF = str(ASSETS_DIR / "franka_panda/panda.urdf")


class PinocchioDynamicsTest(unittest.TestCase):
    """Test cases to validate the manipulator dynamics against Pinocchio's values"""

    @classmethod
    def setUpClass(cls):
        if not PIN_INSTALLED:
            raise unittest.SkipTest("Pinocchio not installed")
        cls.model = pin.buildModelFromUrdf(URDF)
        cls.data = pin.Data(cls.model)
        cls.robot = Manipulator.from_urdf(URDF)
        cls.num_joints = cls.robot.num_joints
        np.random.seed(0)

    def test_mass_matrix(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            dq = np.zeros(self.num_joints)
            pin.forwardKinematics(self.model, self.data, q, dq)
            pin.updateFramePlacements(self.model, self.data)
            M = self.robot.mass_matrix(q)
            Mpin = pin.crba(self.model, self.data, q)
            np.testing.assert_array_almost_equal(M, Mpin, decimal=4)

    def test_nonlinear_effects(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            dq = np.random.rand(self.num_joints)
            pin.forwardKinematics(self.model, self.data, q, dq)
            pin.updateFramePlacements(self.model, self.data)
            bias = pin.nle(self.model, self.data, q, dq)
            G = self.robot.gravity_vector(q)
            C = self.robot.centrifugal_coriolis_vector(q, dq)
            np.testing.assert_array_almost_equal(G + C, bias, decimal=4)


class PybulletDynamicsTest(unittest.TestCase):
    """Test cases to validate the manipulator dynamics against Pybullet's values"""

    @classmethod
    def setUpClass(cls):
        # Setup pybullet environment
        pybullet.connect(pybullet.DIRECT)
        cls.robot_id = pybullet.loadURDF(
            URDF,
            useFixedBase=True,
            flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
        )
        pybullet.setGravity(0, 0, -9.81)
        cls.robot = Manipulator.from_urdf(URDF)
        cls.num_joints = cls.robot.num_joints
        np.random.seed(0)

    @classmethod
    def tearDownClass(cls):
        pybullet.disconnect()

    def test_mass_matrix(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            mass_matrix = pybullet.calculateMassMatrix(self.robot_id, q.tolist())
            M = self.robot.mass_matrix(q)
            np.testing.assert_array_almost_equal(M, mass_matrix, decimal=4)

    def test_gravity_vector(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            zero_velocities = [0.0] * self.num_joints
            zero_accelerations = [0.0] * self.num_joints

            gravity_torques = pybullet.calculateInverseDynamics(
                self.robot_id, q.tolist(), zero_velocities, zero_accelerations
            )
            G = self.robot.gravity_vector(q)
            np.testing.assert_array_almost_equal(G, gravity_torques, decimal=4)

    def test_coriolis_vector(self):
        for _ in range(10):
            q = jnp.array(np.random.rand(self.num_joints))
            qd = jnp.array(np.random.rand(self.num_joints))
            zero_velocities = [0.0] * self.num_joints
            zero_accelerations = [0.0] * self.num_joints

            gravity_torques = pybullet.calculateInverseDynamics(
                self.robot_id, q.tolist(), zero_velocities, zero_accelerations
            )

            coriolis_gravity_torques = pybullet.calculateInverseDynamics(
                self.robot_id, q.tolist(), qd.tolist(), zero_accelerations
            )

            coriolis_torques = np.array(coriolis_gravity_torques) - np.array(
                gravity_torques
            )

            C = self.robot.centrifugal_coriolis_vector(q, qd)
            np.testing.assert_array_almost_equal(C, coriolis_torques, decimal=4)

    def test_jacobian(self):
        for _ in range(10):
            q = jnp.array(np.random.rand(self.num_joints))
            qdot = np.zeros(self.num_joints).tolist()
            qddot = np.zeros(self.num_joints).tolist()
            link_index = self.num_joints - 1
            jv, jw = pybullet.calculateJacobian(
                self.robot_id, link_index, (0, 0, 0), q.tolist(), qdot, qddot
            )
            J_pybullet = np.row_stack([np.array(jv), np.array(jw)])
            J = self.robot.ee_jacobian(q)
            np.testing.assert_array_almost_equal(J, J_pybullet, decimal=4)


if __name__ == "__main__":
    unittest.main()

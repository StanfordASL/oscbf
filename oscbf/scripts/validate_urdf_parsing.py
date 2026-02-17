"""Script to check that the URDF was properly parsed

This will bring up a visualizer in Pybullet and show the transformations between joints
based on the parsed kinematics.

If the URDF was not parsed properly, the lines will not match up with the robot, as
the robot is moved around in the GUI

NOTE: Since reference frames may be coincident with eachother, the number of points and lines
might appear to be lower than expected
"""

import pybullet
import numpy as np

from oscbf.core.manipulator import Manipulator
from oscbf.assets import ASSETS_DIR

URDF = str(ASSETS_DIR / "franka_panda/panda.urdf")
FRANKA_INIT_QPOS = np.array(
    [0.0, -np.pi / 6, 0.0, -3 * np.pi / 4, 0.0, 5 * np.pi / 9, 0.0]
)


def visualize_parsed_tfs(urdf: str, randomize: bool = False):
    pybullet.connect(pybullet.GUI)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME, 1)
    robot = pybullet.loadURDF(
        urdf,
        useFixedBase=True,
        flags=pybullet.URDF_MERGE_FIXED_LINKS | pybullet.URDF_USE_INERTIA_FROM_FILE,
    )
    manipulator = Manipulator.from_urdf(urdf)

    if randomize:
        # Set some random joint angles
        for i in range(manipulator.num_joints):
            pybullet.resetJointState(robot, i, 2 * np.random.rand() - 1)
    else:
        # Use a "good" initial joint configuration
        for i in range(manipulator.num_joints):
            pybullet.resetJointState(robot, i, FRANKA_INIT_QPOS[i])

    num_lines = manipulator.num_joints - 1
    lines = [-1] * num_lines  # init
    joint_pts = -1  # init
    joint_texts = [-1] * manipulator.num_joints  # init
    com_pts = [-1] * manipulator.num_joints  # init
    com_texts = [-1] * manipulator.num_joints  # init
    states = pybullet.getJointStates(robot, range(manipulator.num_joints))
    q = np.array([states[i][0] for i in range(manipulator.num_joints)])

    joint_transforms = manipulator.joint_to_world_transforms(q)

    rgbs = [[1, 0, 0]] * num_lines

    while True:
        states = pybullet.getJointStates(robot, range(manipulator.num_joints))
        q = np.array([states[i][0] for i in range(manipulator.num_joints)])
        joint_transforms = manipulator.joint_to_world_transforms(q)
        link_transforms = manipulator.link_inertial_to_world_transforms(q)
        joint_pts = pybullet.addUserDebugPoints(
            [tf[:3, 3] for tf in joint_transforms],
            [[0, 0, 0]] * manipulator.num_joints,
            10,
            replaceItemUniqueId=joint_pts,
        )
        for i in range(manipulator.num_joints):
            com_pts[i] = pybullet.addUserDebugPoints(
                [link_transforms[i][:3, 3]],
                [[0, 1, 0]],
                10,
                replaceItemUniqueId=com_pts[i],
            )
            com_texts[i] = pybullet.addUserDebugText(
                f"COM {i}",
                link_transforms[i][:3, 3],
                [0, 1, 0],
                replaceItemUniqueId=com_texts[i],
            )
            joint_texts[i] = pybullet.addUserDebugText(
                f"Joint {i}",
                joint_transforms[i][:3, 3],
                [0, 0, 0],
                replaceItemUniqueId=joint_texts[i],
            )
        for i in range(num_lines):
            lines[i] = pybullet.addUserDebugLine(
                joint_transforms[i][:3, 3],
                joint_transforms[i + 1][:3, 3],
                rgbs[i],
                3,
                replaceItemUniqueId=lines[i],
            )
        pybullet.stepSimulation()


def main():
    visualize_parsed_tfs(URDF, randomize=False)


if __name__ == "__main__":
    main()

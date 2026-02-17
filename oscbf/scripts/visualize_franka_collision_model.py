"""Script to visualize the hand-designed collision model of the Franka Panda in Pybullet"""

import pybullet
import numpy as np

import oscbf.core.franka_collision_model as colmodel
from oscbf.core.manipulator import create_transform_numpy, load_panda
from oscbf.utils.visualization import visualize_3D_sphere
from oscbf.assets import ASSETS_DIR

np.random.seed(0)

URDF = str(ASSETS_DIR / "franka_panda/panda.urdf")
FRANKA_INIT_QPOS = np.array(
    [0.0, -np.pi / 6, 0.0, -3 * np.pi / 4, 0.0, 5 * np.pi / 9, 0.0]
)
RANDOMIZE = False


def visualize_collision_model(
    positions,
    radii,
    pairs=None,
    base_position=None,
    base_radius=None,
    base_sc_idxs=None,
):
    pybullet.connect(pybullet.GUI)
    robot = pybullet.loadURDF(
        URDF,
        useFixedBase=True,
        flags=pybullet.URDF_USE_INERTIA_FROM_FILE | pybullet.URDF_MERGE_FIXED_LINKS,
    )
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    manipulator = load_panda()
    for link_idx in range(manipulator.num_joints):
        pybullet.changeVisualShape(robot, link_idx, rgbaColor=(0, 0, 0, 0.5))
    pybullet.changeVisualShape(robot, -1, rgbaColor=(0, 0, 0, 0.5))

    # input("Press Enter to randomize the joints")
    if RANDOMIZE:
        q = np.random.rand(manipulator.num_joints)
    else:
        q = FRANKA_INIT_QPOS
    for i in range(manipulator.num_joints):
        pybullet.resetJointState(robot, i, q[i])
    pybullet.stepSimulation()
    joint_transforms = manipulator.joint_to_world_transforms(q)

    sphere_ids = []
    sphere_positions = []
    # Determine the world-frame positions of the collision geometry
    for i in range(manipulator.num_joints):
        parent_to_world_tf = joint_transforms[i]
        num_collision_spheres = len(positions[i])
        for j in range(num_collision_spheres):
            collision_to_parent_tf = create_transform_numpy(np.eye(3), positions[i][j])
            collision_to_world_tf = parent_to_world_tf @ collision_to_parent_tf
            world_pos = collision_to_world_tf[:3, 3]
            sphere_ids.append(visualize_3D_sphere(world_pos, radii[i][j]))
            sphere_positions.append(world_pos)

    if pairs is not None:
        for pair in pairs:
            i, j = pair
            rgb = (1, 0, 0)
            pybullet.addUserDebugLine(sphere_positions[i], sphere_positions[j], rgb)

    if base_position is not None:
        visualize_3D_sphere(base_position, base_radius)

    if base_sc_idxs is not None:
        for idx in base_sc_idxs:
            rgb = (1, 0, 0)
            pybullet.addUserDebugLine(base_position, sphere_positions[idx], rgb)

    input("Press Enter to exit")


def main(self_collision=False):
    if self_collision:
        visualize_collision_model(
            colmodel.positions_list_sc,
            colmodel.radii_list_sc,
            colmodel.pairs_sc,
            colmodel.base_position,
            colmodel.base_radius,
            colmodel.base_sc_idxs,
        )
    else:
        visualize_collision_model(colmodel.positions_list, colmodel.radii_list)


if __name__ == "__main__":
    main()

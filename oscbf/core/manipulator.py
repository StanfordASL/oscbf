"""Dynamics of a serial manipulator"""

from typing import Tuple, Optional

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np

from oscbf.utils.urdf_parser import parse_urdf
from oscbf.core.franka_collision_model import franka_collision_data, franka_self_collision_data
from oscbf.utils.general_utils import find_assets_dir


def tuplify(arr):
    """Recursively convert a nested structure (for instance, lists, tuples, arrays) to a tuple.
    If arr is not either of these, it returns the original value.

    For instance,
    ```
    tuplify([1, 2, [3, 4]]) == (1, 2, (3, 4))
    tuplify(np.array([[1, 2], [3, 4]])) == ((1, 2), (3, 4))
    tuplify(1) == 1
    tuplify("hello") == "hello"
    ```

    Args:
        arr (Any): A (possibly) nested structure of lists, tuples, arrays, etc.

    Returns:
        Any: All lists/tuples/arrays converted to tuples; other values unchanged
    """
    if isinstance(arr, (list, tuple)):
        return tuple(tuplify(a) for a in arr)
    elif hasattr(arr, "tolist") and callable(arr.tolist):
        # This handles numpy and jax arrays
        return tuplify(arr.tolist())
    else:
        return arr


def create_transform(rotation: ArrayLike, translation: ArrayLike) -> Array:
    """Create a transformation matrix from a rotation matrix and a translation vector

    Args:
        rotation (ArrayLike): Rotation matrix, shape (3, 3)
        translation (ArrayLike): Translation vector, shape (3,)

    Returns:
        Array: Transformation matrix, shape (4, 4)
    """
    return jnp.block(
        [
            [jnp.asarray(rotation), jnp.asarray(translation).reshape(-1, 1)],
            [jnp.array([[0.0, 0.0, 0.0, 1.0]])],
        ]
    )


def create_transform_numpy(rotation: ArrayLike, translation: ArrayLike) -> np.ndarray:
    """Create a transformation matrix from a rotation matrix and a translation vector

    Args:
        rotation (ArrayLike): Rotation matrix, shape (3, 3)
        translation (ArrayLike): Translation vector, shape (3,)

    Returns:
        Array: Transformation matrix, shape (4, 4)
    """
    return np.block(
        [
            [np.asarray(rotation), np.asarray(translation).reshape(-1, 1)],
            [np.array([[0.0, 0.0, 0.0, 1.0]])],
        ]
    )


def revolute_transform(q: float, axis: ArrayLike) -> jax.Array:
    """Create a transformation matrix for a revolute joint (child frame --> parent frame)

    Args:
        q (float): Joint angle
        axis (ArrayLike): Joint axis (in joint frame), shape (3,)

    Returns:
        jax.Array: Transformation matrix, shape (4, 4)
    """
    axis = jnp.asarray(axis)
    axis = axis / jnp.linalg.norm(axis)
    a1, a2, a3 = axis
    c = jnp.cos(q)
    s = jnp.sin(q)
    t = 1 - c
    return jnp.array(
        [
            [t * a1 * a1 + c, t * a1 * a2 - s * a3, t * a1 * a3 + s * a2, 0.0],
            [t * a1 * a2 + s * a3, t * a2 * a2 + c, t * a2 * a3 - s * a1, 0.0],
            [t * a1 * a3 - s * a2, t * a2 * a3 + s * a1, t * a3 * a3 + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def prismatic_transform(q: float, axis: ArrayLike) -> jax.Array:
    """Create a transformation matrix for a prismatic joint (child frame --> parent frame)

    Args:
        q (float): Joint position
        axis (ArrayLike): Joint axis (in joint frame), shape (3,)

    Returns:
        jax.Array: Transformation matrix, shape (4, 4)
    """
    axis = jnp.asarray(axis)
    translation = q * axis / jnp.linalg.norm(axis)
    return jnp.array(
        [
            [1.0, 0.0, 0.0, translation[0]],
            [0.0, 1.0, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def transform_point(transform, point):
    transform = jnp.asarray(transform)
    point = jnp.asarray(point)
    assert transform.shape == (4, 4)
    assert point.shape == (3,)
    return (transform @ jnp.concatenate([point, 1.0]))[:3]


def transform_points(transform, points):
    transform = jnp.asarray(transform)
    points = jnp.asarray(points)
    assert transform.shape == (4, 4)
    assert points.shape[1] == 3
    return (jnp.hstack([points, jnp.ones((points.shape[0], 1))]) @ transform.T)[:, :3]


def joint_transform(
    joint_pos: float, joint_axis: ArrayLike, joint_type: float
) -> Array:
    """Create a transformation matrix for a joint (child frame --> parent frame) based on the joint type

    Args:
        joint_pos (float): Joint position
        joint_axis (ArrayLike): Joint axis (in joint frame), shape (3,)
        joint_type (float): Joint type (0 for revolute, 1 for prismatic)

    Returns:
        Array: Transformation matrix, shape (4, 4)
    """
    return (1.0 - joint_type) * revolute_transform(
        joint_pos, joint_axis
    ) + joint_type * prismatic_transform(joint_pos, joint_axis)


@jax.tree_util.register_static
class Manipulator:
    """Manipulator kinematics and dynamics

    Note: the main constructor for this class is via the `from_urdf` classmethod.
    See this method for more details
    """

    def __init__(
        self,
        num_joints: int,
        joint_types: tuple,
        joint_lower_limits: tuple,
        joint_upper_limits: tuple,
        joint_max_forces: tuple,
        joint_max_velocities: tuple,
        joint_axes: tuple,
        joint_parent_frame_positions: tuple,
        joint_parent_frame_rotations: tuple,
        link_masses: tuple,
        link_local_inertias: tuple,
        link_local_inertia_positions: tuple,
        link_local_inertia_rotations: tuple,
        base_pos: tuple,
        base_orn: tuple,
        ee_offset: tuple,
        collision_positions: tuple,
        collision_radii: tuple,
        self_collision_positions: tuple,
        self_collision_radii: tuple,
        self_collision_pairs: tuple,
    ):
        # fmt: off
        assert isinstance(num_joints, int)
        assert isinstance(joint_types, tuple) and len(joint_types) == num_joints
        assert isinstance(joint_lower_limits, tuple) and len(joint_lower_limits) == num_joints
        assert isinstance(joint_upper_limits, tuple) and len(joint_upper_limits) == num_joints
        assert isinstance(joint_max_forces, tuple) and len(joint_max_forces) == num_joints
        assert isinstance(joint_max_velocities, tuple) and len(joint_max_velocities) == num_joints
        assert isinstance(joint_axes, tuple) and len(joint_axes) == num_joints
        assert isinstance(joint_parent_frame_positions, tuple) and len(joint_parent_frame_positions) == num_joints
        assert isinstance(joint_parent_frame_rotations, tuple) and len(joint_parent_frame_rotations) == num_joints
        assert isinstance(link_masses, tuple) and len(link_masses) == num_joints
        assert isinstance(link_local_inertias, tuple) and len(link_local_inertias) == num_joints
        assert isinstance(link_local_inertia_positions, tuple) and len(link_local_inertia_positions) == num_joints
        assert isinstance(link_local_inertia_rotations, tuple) and len(link_local_inertia_rotations) == num_joints
        # assert isinstance(base_pos, tuple) and len(base_pos) == 3
        # assert isinstance(base_orn, tuple) and len(base_orn) == 4
        assert isinstance(ee_offset, tuple) and len(ee_offset) == 4  # Transformation matrix
        assert isinstance(collision_positions, tuple)
        assert isinstance(collision_radii, tuple)
        assert isinstance(self_collision_positions, tuple)
        assert isinstance(self_collision_radii, tuple)
        assert isinstance(self_collision_pairs, tuple)
        # fmt: on

        self.num_joints = num_joints
        self.joint_types = joint_types
        self.joint_lower_limits = joint_lower_limits
        self.joint_upper_limits = joint_upper_limits
        self.joint_max_forces = joint_max_forces
        self.joint_max_velocities = joint_max_velocities
        self.joint_axes = joint_axes
        self.joint_parent_frame_positions = joint_parent_frame_positions
        self.joint_parent_frame_rotations = joint_parent_frame_rotations
        self.link_masses = link_masses
        self.link_local_inertias = link_local_inertias
        self.link_local_inertia_positions = link_local_inertia_positions
        self.link_local_inertia_rotations = link_local_inertia_rotations
        # NOTE: Assume a fixed base, in which case we don't need inertial info of the base
        self.base_pos = base_pos
        self.base_orn = base_orn
        self.ee_offset = ee_offset
        self.collision_positions = collision_positions
        self.collision_radii = collision_radii
        self.self_collision_positions = self_collision_positions
        self.self_collision_radii = self_collision_radii
        self.self_collision_pairs = self_collision_pairs
        # self.num_collision_pts_per_link = tuple(len(pos) for pos in collision_positions)
        # self.num_collision_points = sum(self.num_collision_pts_per_link)
        self.has_collision_data = len(collision_positions) > 0
        self.has_self_collision_data = len(self_collision_positions) > 0
        self.joint_to_prev_joint_tfs = tuplify(
            [
                create_transform_numpy(rot, trans)
                for rot, trans in zip(
                    self.joint_parent_frame_rotations, self.joint_parent_frame_positions
                )
            ]
        )
        self.link_com_to_prev_joint_tfs = tuplify(
            [
                create_transform_numpy(rot, trans)
                for rot, trans in zip(
                    self.link_local_inertia_rotations, self.link_local_inertia_positions
                )
            ]
        )
        # Set up padded collision positions and radii for vmapping with static shape
        (
            self.padded_collision_positions,
            self.padded_collision_radii,
            self.collision_slice_indices,
        ) = self._process_collision_data(collision_positions, collision_radii)

    @classmethod
    def from_urdf(
        cls,
        urdf_filename: str,
        ee_offset: Optional[ArrayLike] = None,
        collision_data: Optional[dict] = None,
        self_collision_data: Optional[dict] = None,
    ) -> "Manipulator":
        """Construct a Manipulator object from a parsed URDF file

        Note: the URDF should only contain joints that are part of the kinematic chain and thus
        are going to be actively controlled. All other joints should be set to "fixed" so that their
        kinematics and dynamics properties are merged into the chain.

        Args:
            urdf_filename (str): Path to the URDF file
            ee_offset (ArrayLike, optional): End-effector / tool-center-point transformation
                from the last joint frame, shape (4, 4). Defaults to None.
            collision_data (dict, optional): Collision geometry for each link, stored as a dictionary
                where data["positions"] => list of sphere center points in each link frame, and
                data["radii"] => list of sphere radii for each body. Defaults to None.
            self_collision_data (dict, optional): Self collision geometry for each link, stored as a dictionary
                where data["positions"] => list of sphere center points in each link frame,
                data["radii"] => list of sphere radii for each body, and
                data["pairs"] => list of pairs of collision sphere indices to consider. Defaults to None.

        Returns:
            Manipulator: The manipulator object constructed from the URDF
        """

        data = parse_urdf(urdf_filename)
        data = {k: tuplify(v) for k, v in data.items()}

        assert isinstance(collision_data, dict) or collision_data is None
        if isinstance(collision_data, dict):
            collision_positions = collision_data["positions"]
            collision_radii = collision_data["radii"]
        else:
            collision_positions = ()
            collision_radii = ()

        assert isinstance(self_collision_data, dict) or self_collision_data is None
        if isinstance(self_collision_data, dict):
            self_collision_positions = self_collision_data["positions"]
            self_collision_radii = self_collision_data["radii"]
            self_collision_pairs = self_collision_data["pairs"]
        else:
            self_collision_positions = ()
            self_collision_radii = ()
            self_collision_pairs = ()

        if ee_offset is None:
            ee_offset = tuplify(np.eye(4))
        else:
            ee_offset = np.asarray(ee_offset)
            assert ee_offset.shape == (4, 4)
            ee_offset = tuplify(ee_offset)

        return cls(
            num_joints=data["num_joints"],
            joint_types=data["joint_types"],
            joint_lower_limits=data["joint_lower_limits"],
            joint_upper_limits=data["joint_upper_limits"],
            joint_max_forces=data["joint_max_forces"],
            joint_max_velocities=data["joint_max_velocities"],
            joint_axes=data["joint_axes"],
            joint_parent_frame_positions=data["joint_parent_frame_positions"],
            joint_parent_frame_rotations=data["joint_parent_frame_rotations"],
            link_masses=data["link_masses"],
            link_local_inertias=data["link_local_inertias"],
            link_local_inertia_positions=data["link_local_inertia_positions"],
            link_local_inertia_rotations=data["link_local_inertia_rotations"],
            # TEMP: ignore base data
            base_pos=None,  # tuple(data["base_pos"]),
            base_orn=None,  # tuple(data["base_orn"]),
            ee_offset=ee_offset,
            collision_positions=collision_positions,
            collision_radii=collision_radii,
            self_collision_positions=self_collision_positions,
            self_collision_radii=self_collision_radii,
            self_collision_pairs=self_collision_pairs,
        )

    def _process_collision_data(self, positions: tuple, radii: tuple) -> Tuple[tuple, tuple, tuple]:
        """Helper function: Sets up a padded representation of the collision sphere positions and
        radii for vmapping with uniform shape

        This should only be called once upon initialization

        Args:
            positions (tuple): Collision sphere locations for each link. Tuple of len (num_links)
                where each entry contains a set of points (length 3) for that link
            radii (tuple): Collision sphere radii for each link. Tuple of len (num_links)
                where each entry contains a set of radii for that link

        Returns:
            Tuple[tuple, tuple, tuple]:
                padded_positions: Positions, shape (num_joints, max_spheres_per_link, 3)
                padded_radii: Radii, shape (num_joints, max_spheres_per_link)
                slice_indices: Indices of the *flattened* padded position and radii to select,
                    corresponding to the non-padded data
        """
        assert isinstance(positions, tuple)
        assert isinstance(radii, tuple)
        if len(positions) == 0 or len(radii) == 0:
            return (), (), ()
        sphere_counts = tuple(len(rs) for rs in radii)
        max_spheres_per_link = max(sphere_counts)
        padded_positions = np.zeros((self.num_joints, max_spheres_per_link, 3))
        padded_radii = np.zeros((self.num_joints, max_spheres_per_link))
        for link_idx in range(self.num_joints):
            for sphere_idx in range(sphere_counts[link_idx]):
                padded_positions[link_idx, sphere_idx] = positions[link_idx][sphere_idx]
                padded_radii[link_idx, sphere_idx] = radii[link_idx][sphere_idx]
        padded_positions = tuplify(padded_positions)
        padded_radii = tuplify(padded_radii)
        # mask: (num_joints, max_spheres) - True for non-padded spheres
        sphere_mask = np.arange(max_spheres_per_link) < np.asarray(sphere_counts)[:, None]
        slice_indices = tuplify(np.flatnonzero(sphere_mask.flatten()))
        return padded_positions, padded_radii, slice_indices

    def joint_to_world_transforms(self, q: Array) -> Array:
        """Computes the transformation matrices for all joints (Joint frame --> world frame)

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Transformation matrices, shape (num_joints, 4, 4)
        """
        # Convert static data to jax arrays
        joint_axes = jnp.asarray(self.joint_axes)
        joint_types = jnp.asarray(self.joint_types)
        joint_to_prev_joint_tfs = jnp.asarray(self.joint_to_prev_joint_tfs)
        # Calculate each joint's transformation matrix
        transforms = jax.vmap(joint_transform)(
            q,
            joint_axes,
            joint_types,
        )

        # Multiply the transform by its corresponding link offset
        transforms = joint_to_prev_joint_tfs @ transforms

        # Calculate the cumulative product of the transformations
        cumulative_transforms = jax.lax.associative_scan(
            jnp.matmul, transforms, reverse=False, axis=0
        )

        return cumulative_transforms

    def link_inertial_to_world_transforms(self, q: Array) -> Array:
        """Compute the transformation matrices for all link inertial frames (link inertial frame --> world frame)

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Transformation matrices, shape (num_joints, 4, 4)
        """
        # Convert static data to jax arrays
        link_com_to_prev_joint_tfs = jnp.asarray(self.link_com_to_prev_joint_tfs)

        joint_transforms = self.joint_to_world_transforms(q)
        # Multiply the transform by its corresponding link offset
        transforms = joint_transforms @ link_com_to_prev_joint_tfs
        return transforms

    @jax.jit
    def ee_transform(self, q: Array) -> Array:
        """Compute the transformation matrix of the end effector (EE frame --> world frame)

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Transformation matrix, shape (4, 4)
        """
        transforms = self.joint_to_world_transforms(q)
        return self._ee_transform(transforms)

    def _ee_transform(self, joint_transforms: Array) -> Array:
        """Helper function: Compute EE transform given joint transforms"""
        return joint_transforms[-1] @ jnp.asarray(self.ee_offset)

    def ee_position(self, q: Array) -> Array:
        """Compute the position of the end effector in world frame

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: End-effector position, shape (3,)
        """
        return self.ee_transform(q)[:3, 3]

    def ee_rotation(self, q: Array) -> Array:
        """Compute the rotation matrix of the end effector in world frame

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: End-effector rotation matrix, shape (3, 3)
        """
        return self.ee_transform(q)[:3, :3]

    @jax.jit
    def ee_jacobian(self, q: Array) -> Array:
        """Compute the basic Jacobian (J_0) of the end effector given the joint configuration

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Jacobian, shape (6, num_joints). The first 3 rows are the linear Jacobian,
                and the last 3 rows are the angular Jacobian
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._ee_jacobian(joint_transforms)

    def _ee_jacobian(self, joint_transforms: Array) -> Array:
        """Helper function: Compute EE jacobian given joint transforms"""
        # Convert static data to jax arrays
        joint_types = jnp.asarray(self.joint_types)
        ee_offset = jnp.asarray(self.ee_offset)

        # Prismatic/revolute masks
        epsilon = joint_types.reshape(-1, 1)
        epsilon_bar = 1.0 - epsilon

        ee_pos = (joint_transforms[-1] @ ee_offset)[:3, 3]

        # Positions of all joints in world frame. Shape (num_joints, 3)
        joint_pos = joint_transforms[:, :3, 3]

        # Axes of all joints in world frame. Shape (num_joints, 3)
        joint_axes_world = (
            joint_transforms[:, :3, :3] @ np.asarray(self.joint_axes)[:, :, np.newaxis]
        ).squeeze(axis=2)

        # Position of EE, with respect to joint j. Shape (num_joints, 3).
        ee_wrt_joints = ee_pos[jnp.newaxis, :] - joint_pos

        # Cross products between joint axis j and ee position with respect to joint j.
        # Shape (num_joints, 3)
        lever_arms = jnp.cross(joint_axes_world, ee_wrt_joints)

        Jv = (epsilon * joint_axes_world + epsilon_bar * lever_arms).T
        Jw = (epsilon_bar * joint_axes_world).T
        return jnp.vstack([Jv, Jw])

    @jax.jit
    def ee_jacobian_derivative(self, q: Array, qd: Array) -> Array:
        """Compute the time derivative of the end effector Jacobian

        Args:
            q (Array): Joint positions, shape (num_joints,)
            qd (Array): Joint velocities, shape (num_joints,)

        Returns:
            Array: Time derivative of the end effector Jacobian, shape (6, num_joints)
        """
        return jax.jvp(self.ee_jacobian, (q,), (qd,))[1]

    def _link_com_positions(self, joint_transforms: Array) -> Array:
        """Helper function: Compute the positions of all link COMs in world frame, given the joint transforms"""
        # Convert static data to jax arrays
        link_local_inertia_positions = jnp.asarray(self.link_local_inertia_positions)
        # Determine the positions of the link COMs in world frame. Shape (num_joints, 3)
        homogeneous_pos = jnp.column_stack(
            [jnp.array(link_local_inertia_positions), jnp.ones(self.num_joints)]
        )
        link_com_pos = (joint_transforms @ homogeneous_pos[:, :, jnp.newaxis])[:, :3, 0]
        return link_com_pos

    def link_com_positions(self, q: Array) -> Array:
        """Compute the positions of all link COMs in world frame

        Args:
            q (Array): Joint angles, shape (num_joints,)

        Returns:
            Array: Link COM positions in world frame, shape (num_links, 3)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._link_com_positions(joint_transforms)

    @jax.jit
    def link_collision_data(self, q: Array) -> Array:
        """Compute collision data for all links given the joint configuration

        Collision data includes the positions of all collision points (in world frame)
        and their radii, stacked together in an array

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Collision data, shape (num_collision_points, 4)
                The first three columns are the xyz world-frame positions of the collision points,
                and the fourth column is the radii
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._link_collision_data(joint_transforms)

    def _link_collision_data(self, joint_transforms: Array) -> Array:
        """Helper function: Compute the collision data for all links given the joint transforms"""
        if not self.has_collision_data:
            return jnp.array([])

        # Convert static data to jax arrays
        padded_collision_positions = jnp.asarray(self.padded_collision_positions)
        padded_collision_radii = jnp.asarray(self.padded_collision_radii)
        collision_slice_indices = jnp.asarray(self.collision_slice_indices)

        # Compute collision body positions in world frame
        transformed_pts_padded = jax.vmap(transform_points)(
            joint_transforms, padded_collision_positions
        )  # Shape (num_joints, max_spheres, 3)

        # Flatten and select only the non-padded collision data
        # Flat points shape (num_joints * max_spheres, 3),
        # Flat radii shape (num_joints * max_spheres, 1)
        all_pts_flat = transformed_pts_padded.reshape(-1, 3)
        all_radii_flat = padded_collision_radii.reshape(-1, 1)
        pts_unpadded = all_pts_flat[collision_slice_indices]
        radii_unpadded = all_radii_flat[collision_slice_indices]

        return jnp.hstack([pts_unpadded, radii_unpadded])

    @jax.jit
    def link_self_collision_data(self, q: Array) -> Array:
        """Compute collision data for all links given the joint configuration

        Collision data includes the positions of all collision points (in world frame)
        and their radii, stacked together in an array

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Array: Collision data, shape (num_collision_points, 4)
                The first three columns are the xyz world-frame positions of the collision points,
                and the fourth column is the radii
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._link_self_collision_data(joint_transforms)

    # TODO: use vmap
    def _link_self_collision_data(self, joint_transforms: Array) -> Array:
        """Helper function: Compute the self collision data for all links given the joint transforms"""
        if not self.has_self_collision_data:
            return jnp.array([])

        pts = []
        radii = []
        for i in range(self.num_joints):
            if len(self.self_collision_positions[i]) > 0:
                pts.append(
                    transform_points(joint_transforms[i], jnp.asarray(self.self_collision_positions[i]))
                )
                radii.append(jnp.asarray(self.self_collision_radii[i]))
        pts = jnp.vstack(pts)
        radii = jnp.concatenate(radii).reshape(-1, 1)
        return jnp.hstack([pts, radii])

    def _get_linear_jacobians_transposed(self, joint_transforms: Array) -> Array:
        """Helper function: Compute an array containing the linear jacobians Jv for every link

        Note the transposed shape: (num_joints, num_joints, 3), rather than (num_joints, 3, num_joints).
        This is slightly easier for vectorized operations

        Args:
            transforms (Array): Transformation matrices for every joint, shape (num_joints, 4, 4)

        Returns:
            Array: Linear jacobians for every link, shape (num_joints, num_joints, 3)
        """
        # Convert static data to jax arrays
        link_local_inertia_positions = jnp.asarray(self.link_local_inertia_positions)
        joint_types = jnp.asarray(self.joint_types)

        # Prismatic/revolute masks
        epsilon = joint_types.reshape(-1, 1)
        epsilon_bar = 1.0 - epsilon

        # TODO: Use _link_com_positions method instead? Less repeated code
        # Determine the positions of the link COMs in world frame. Shape (num_joints, 3)
        homogeneous_pos = jnp.column_stack(
            [jnp.array(link_local_inertia_positions), jnp.ones(self.num_joints)]
        )
        link_com_pos = (joint_transforms @ homogeneous_pos[:, :, jnp.newaxis])[:, :3, 0]

        # Positions of all joints in world frame. Shape (num_joints, 3)
        joint_pos = joint_transforms[:, :3, 3]

        # Z axes of all joints in world frame. Shape (num_joints, 3)
        joint_axes_world_frame = (
            joint_transforms[:, :3, :3] @ np.asarray(self.joint_axes)[:, :, np.newaxis]
        ).squeeze(axis=2)

        # Positions of link COM i, with respect to joint j. Shape (num_joints, num_joints, 3).
        link_com_wrt_joints = (
            link_com_pos[:, jnp.newaxis, :] - joint_pos[jnp.newaxis, :, :]
        )

        # Cross products between joint axis j and link COM i's position with respect to joint j.
        # Shape (num_joints, num_joints, 3)
        stacked_axes = jnp.tile(joint_axes_world_frame, (self.num_joints, 1, 1))
        lever_arms = jnp.cross(stacked_axes, link_com_wrt_joints)

        # Note: The jacobian associated with link i only depends on the joints j <= i
        # So, use a lower triangular mask to only include these components
        mask_matrix = jnp.tril(jnp.ones((self.num_joints, self.num_joints)))
        mask_3d = mask_matrix[:, :, jnp.newaxis]
        return mask_3d * (epsilon * stacked_axes + epsilon_bar * lever_arms)

    def _get_angular_jacobians_transposed(self, joint_transforms: Array) -> Array:
        """Helper function: Compute an array containing the angular jacobians Jw for every link

        Note the transposed shape: (num_joints, num_joints, 3), rather than (num_joints, 3, num_joints).
        This is slightly easier for vectorized operations

        Args:
            transforms (Array): Transformation matrices for every joint, shape (num_joints, 4, 4)

        Returns:
            Array: Angular jacobians for every link, shape (num_joints, num_joints, 3)
        """
        # Convert static data to jax arrays
        joint_types = jnp.asarray(self.joint_types)

        # Prismatic/revolute masks
        epsilon = joint_types.reshape(-1, 1)
        epsilon_bar = 1.0 - epsilon

        joint_axes_world_frame = (
            joint_transforms[:, :3, :3] @ np.asarray(self.joint_axes)[:, :, np.newaxis]
        ).squeeze(axis=2)

        stacked_axes = jnp.tile(joint_axes_world_frame, (self.num_joints, 1, 1))
        mask_matrix = jnp.tril(jnp.ones((self.num_joints, self.num_joints)))
        mask_3d = mask_matrix[:, :, jnp.newaxis]
        return mask_3d * (epsilon_bar * stacked_axes)

    @jax.jit
    def mass_matrix(self, q: Array) -> Array:
        """Compute the mass matrix for a given joint configuration

        Args:
            q (Array): Array of joint angles, shape (num_joints,)

        Returns:
            Array: The mass matrix, shape (num_joints, num_joints)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._mass_matrix(joint_transforms)

    def _mass_matrix(self, joint_transforms: Array) -> Array:
        """Helper function: Compute mass matrix given joint transforms"""
        # Convert static data to jax arrays
        link_masses = jnp.asarray(self.link_masses)
        inertia_mats = jnp.asarray(self.link_local_inertias)
        link_com_to_prev_joint_tfs = jnp.asarray(self.link_com_to_prev_joint_tfs)

        Jv_Ts = self._get_linear_jacobians_transposed(joint_transforms)
        Jw_Ts = self._get_angular_jacobians_transposed(joint_transforms)

        # Linear jacobian component of the mass matrix
        M_v = jnp.sum(
            link_masses[:, jnp.newaxis, jnp.newaxis] * Jv_Ts @ Jv_Ts.transpose(0, 2, 1),
            axis=0,
        )

        # Angular jacobian component of the mass matrix
        link_com_transforms = joint_transforms @ link_com_to_prev_joint_tfs
        link_com_rotations = link_com_transforms[:, :3, :3]
        inertia_mats_world_frame = (
            link_com_rotations @ inertia_mats @ link_com_rotations.transpose(0, 2, 1)
        )
        M_w = jnp.sum(
            Jw_Ts @ inertia_mats_world_frame @ Jw_Ts.transpose(0, 2, 1),
            axis=0,
        )

        return M_v + M_w

    @jax.jit
    def gravity_vector(self, q: Array) -> Array:
        """Compute the gravity vector for a given joint configuration

        Args:
            q (Array): Array of joint angles, shape (num_joints,)

        Returns:
            Array: The gravity vector, shape (num_joints,)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._gravity_vector(joint_transforms)

    def _gravity_vector(self, joint_transforms: Array) -> Array:
        """Helper function: Compute gravity vector given joint transforms"""
        # Convert static data to jax arrays
        link_masses = jnp.asarray(self.link_masses)
        g = jnp.array([0.0, 0.0, -9.81])
        Jv_Ts = self._get_linear_jacobians_transposed(joint_transforms)
        return -jnp.sum(link_masses[:, jnp.newaxis, jnp.newaxis] * Jv_Ts @ g, axis=0)

    @jax.jit
    def centrifugal_coriolis_vector(self, q: Array, qd: Array) -> Array:
        """Compute the centrifugal and coriolis vector for a given joint configuration

        Args:
            q (Array): Array of joint angles, shape (num_joints,)
            qd (Array): Array of joint velocities, shape (num_joints,)

        Returns:
            Array: The centrifugal and coriolis vector, shape (num_joints,)
        """
        # JVP gives us d(M@qd)/dq_i * qd_i
        jvp_result = jax.jvp(lambda q_: self.mass_matrix(q_) @ qd, (q,), (qd,))[1]
        # VJP gives us qd.T @ dM/dq_i * qd_i
        vjp_fun = jax.vjp(lambda q_: qd.T @ self.mass_matrix(q_), q)[1]
        vjp_result = vjp_fun(qd)[0]
        return jvp_result - 0.5 * vjp_result

    @jax.jit
    def torque_control_matrices(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """Compute the matrices required for operational space torque control
        with just a single evaluation of the kinematics

        Args:
            q (Array): Joint positions, shape (num_joints,)
            qd (Array): Joint velocities, shape (num_joints,)

        Returns:
            Tuple[Array, Array, Array, Array, Array, Array]:
                M: Mass matrix, shape (num_joints, num_joints)
                M_inv: Inverse of the mass matrix, shape (num_joints, num_joints)
                G: Gravity vector, shape (num_joints,)
                C: Centrifugal/coriolis vector, shape (num_joints,)
                J: End effector basic Jacobian, shape (6, num_joints)
                T: End effector transformation matrix, shape (4, 4)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        M = self._mass_matrix(joint_transforms)
        M_inv = jnp.linalg.inv(M)
        G = self._gravity_vector(joint_transforms)
        C = self.centrifugal_coriolis_vector(q, qd)
        J = self._ee_jacobian(joint_transforms)
        T = self._ee_transform(joint_transforms)
        return M, M_inv, G, C, J, T

    @jax.jit
    def velocity_control_matrices(self, q: Array) -> Tuple[Array, Array]:
        """Compute the matrices required for operational space velocity control
        with just a single evaluation of the kinematics

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Tuple[Array, Array]:
                J: End effector basic Jacobian, shape (6, num_joints)
                T: End effector transformation matrix, shape (4, 4)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        J = self._ee_jacobian(joint_transforms)
        T = self._ee_transform(joint_transforms)
        return J, T

    @jax.jit
    def dynamically_consistent_velocity_control_matrices(
        self, q: Array
    ) -> Tuple[Array, Array, Array]:
        """Compute the matrices required for operational space velocity control
        with just a single evaluation of the kinematics.

        This version also returns the inverse of the mass matrix, which is required
        to construct the dynamically-consistent generalized Jacobian inverse

        Args:
            q (Array): Joint positions, shape (num_joints,)

        Returns:
            Tuple[Array, Array, Array]:
                M_inv: Inverse of the mass matrix, shape (num_joints, num_joints)
                J: End effector basic Jacobian, shape (6, num_joints)
                T: End effector transformation matrix, shape (4, 4)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        J = self._ee_jacobian(joint_transforms)
        T = self._ee_transform(joint_transforms)
        M = self._mass_matrix(joint_transforms)
        M_inv = jnp.linalg.inv(M)
        return M_inv, J, T


def load_panda() -> Manipulator:
    """Create a Manipulator object for the Franka Panda"""

    return Manipulator.from_urdf(
        find_assets_dir() + "franka_panda/panda.urdf",
        ee_offset=np.block(
            [
                [np.eye(3), np.reshape(np.array([0.0, 0.0, 0.216]), (-1, 1))],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        collision_data=franka_collision_data,
        self_collision_data=franka_self_collision_data,
    )


def main():
    # Quick validation that the manipulator class works
    robot = load_panda()
    q = jnp.zeros(robot.num_joints)
    qdot = 0.1 * jnp.ones(robot.num_joints)
    np.set_printoptions(precision=3, suppress=True)
    print("\nTesting Franka Panda:")
    print("\nMass Matrix: ")
    print(robot.mass_matrix(q))
    print("\nGravity Vector:")
    print(robot.gravity_vector(q))
    print("\nCentrifugal/Coriolis Vector:")
    print(robot.centrifugal_coriolis_vector(q, qdot))
    print("\nEE Jacobian:")
    print(robot.ee_jacobian(q))
    print("\nEE Jacobian Derivative:")
    print(robot.ee_jacobian_derivative(q, qdot))


if __name__ == "__main__":
    main()

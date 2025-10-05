"""Creating a link collision model for the Franka with a series of spheres of various radii"""

# All positions are in link frame, not link COM frame

link_1_pos = (
    (0, 0, -0.15),
    (0, -0.065, 0),
)
link_1_radii = (
    0.06,
    0.06,
)

link_2_pos = (
    (0, 0, 0.065),
    (0, -0.14, 0),
)
link_2_radii = (
    0.06,
    0.06,
)

link_3_pos = (
    (0, 0, -0.065),
    (0.08, 0.065, 0),
)
link_3_radii = (
    0.06,
    0.055,
)

link_4_pos = (
    (0, 0, 0.065),
    (-0.08, 0.08, 0),
)
link_4_radii = (
    0.055,
    0.055,
)

link_5_pos = (
    (0, 0, -0.23),
    (0, 0.06, -0.18),
    (0, 0.08, -0.125),
    (0, 0.09, -0.075),
    (0, 0.08, 0),
)
link_5_radii = (
    0.06,
    0.04,
    0.025,
    0.025,
    0.055,
)

link_6_pos = (
    (0, 0, 0.01),
    (0.08, 0.035, 0),
    # (0.08, -0.02, 0),
)
link_6_radii = (
    0.05,
    0.05,
    # 0.05,
)

link_7_pos = (
    (0, 0, 0.08),
    (0.04, 0.04, 0.09),
    (0.055, 0.055, 0.15),
    (-0.055, -0.055, 0.15),
    (-0.055, -0.055, 0.11),
    (0, 0, 0.20),
)
link_7_radii = (
    0.05,
    0.04,
    0.03,
    0.03,
    0.03,
    0.02,
)

positions_list = (
    link_1_pos,
    link_2_pos,
    link_3_pos,
    link_4_pos,
    link_5_pos,
    link_6_pos,
    link_7_pos,
)
radii_list = (
    link_1_radii,
    link_2_radii,
    link_3_radii,
    link_4_radii,
    link_5_radii,
    link_6_radii,
    link_7_radii,
)

franka_collision_data = {"positions": positions_list, "radii": radii_list}

##### Self collisions #####

# Note: this is a very approximate model with only a small subset of the full geometry.
# However, these few collision pairs are the most relevant for the Franka, and
# tabletop-like tasks in a standard configuration

link_1_pos_sc = (
    (0, 0, -0.13), # 0
    (0.0, -0.05, 0.0),  # 1
)
link_1_radii_sc = (
    0.085,
    0.085,
)

link_2_pos_sc = ((0, 0, 0.05),)  # 2
link_2_radii_sc = (0.085,)

link_3_pos_sc = ()
link_3_radii_sc = ()

link_4_pos_sc = ()
link_4_radii_sc = ()

link_5_pos_sc = (
    (0, 0, -0.23),  # 3
    (0, 0.09, -0.075),  # 4
    (0, 0.08, 0),  # 5
)
link_5_radii_sc = (
    0.08,
    0.04,
    0.065,
)

link_6_pos_sc = ()
link_6_radii_sc = ()

link_7_pos_sc = ((0, 0, 0.12),)  # 6
link_7_radii_sc = (0.12,)

positions_list_sc = (
    link_1_pos_sc,
    link_2_pos_sc,
    link_3_pos_sc,
    link_4_pos_sc,
    link_5_pos_sc,
    link_6_pos_sc,
    link_7_pos_sc,
)
radii_list_sc = (
    link_1_radii_sc,
    link_2_radii_sc,
    link_3_radii_sc,
    link_4_radii_sc,
    link_5_radii_sc,
    link_6_radii_sc,
    link_7_radii_sc,
)
pairs_sc = (
    (0, 6),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 6),
    (3, 6),
)

franka_self_collision_data = {
    "positions": positions_list_sc,
    "radii": radii_list_sc,
    "pairs": pairs_sc,
}

base_position = (-0.04, 0.0, 0.05)
base_radius = 0.135
base_sc_idxs = (6,)

base_self_collision_data = {
    "position": base_position,
    "radius": base_radius,
    "indices": base_sc_idxs,
}
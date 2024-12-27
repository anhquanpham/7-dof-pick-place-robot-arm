import sys
import numpy as np
from copy import deepcopy
import numpy as np
from math import sin,cos,pi
import time

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

from lib.calculateFK import FK
from lib.IK_position_null import IK


def trans(d):
    return np.array([
        [ 1, 0, 0, d[0] ],
        [ 0, 1, 0, d[1] ],
        [ 0, 0, 1, d[2] ],
        [ 0, 0, 0, 1    ],
    ])

def roll(a):
    return np.array([
        [ 1,     0,       0,  0 ],
        [ 0, cos(a), -sin(a), 0 ],
        [ 0, sin(a),  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def pitch(a):
    return np.array([
        [ cos(a), 0, -sin(a), 0 ],
        [      0, 1,       0, 0 ],
        [ sin(a), 0,  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def yaw(a):
    return np.array([
        [ cos(a), -sin(a), 0, 0 ],
        [ sin(a),  cos(a), 0, 0 ],
        [      0,       0, 1, 0 ],
        [      0,       0, 0, 1 ],
    ])

def transform(d,rpy):
    return trans(d) @ roll(rpy[0]) @ pitch(rpy[1]) @ yaw(rpy[2])


def swap(a,b):
    #swap any two variables
    inter = 0
    inter = a
    a=b
    b=inter
    #print(a,b)
    return a,b

def z_swap(a,eps):
    #input: a is rotation matrix, eps is acceptable error
    #output: transformed rotation matrix
    column_no=10
    for i in range(3):
        #check which column has [0,0,1]
        if np.abs(abs(a[2,i])-1)<eps:
            column_no=i
        #swap the column with last column
    for i in range(3):
        a[i,2],a[i,column_no]=swap(a[i,2],a[i,column_no])
    if a[0,0]*a[1,1]<0:
        for i in range(3):
            a[i,0],a[i,1]=swap(a[i,0],a[i,1])
    return a

def filter_static_blocks(detections):
    """Filters detections to only include static blocks."""
    return [detection for detection in detections if "_static" in detection[0]]

def adjust_rotation_for_downward_z(T_b_approach):
    """
    Adjusts the rotation matrix in T_b_approach so that the z-axis points directly downward (-z).

    Parameters:
        T_b_approach (np.array): 4x4 transformation matrix to adjust.

    Returns:
        np.array: Adjusted 4x4 transformation matrix with z-axis aligned downward.
    """
    def rotation_matrix_from_axis_angle(axis, angle):
        """Compute the rotation matrix from an axis and angle using Rodrigues' formula."""
        axis = axis / np.linalg.norm(axis)  # Normalize the axis
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        one_minus_cos = 1 - cos_theta

        # Cross-product matrix of the axis
        x, y, z = axis
        cross_mat = np.array([[0, -z, y],
                              [z, 0, -x],
                              [-y, x, 0]])

        # Rodrigues' formula
        return cos_theta * np.eye(3) + sin_theta * cross_mat + one_minus_cos * np.outer(axis, axis)

    original_rotation = T_b_approach[:3, :3]
    current_z = original_rotation[:, 2]  # Extract the current z-axis

    target_z = np.array([0, 0, -1])  # Desired z-axis direction
    if np.allclose(current_z, target_z, atol=1e-6):
        print("Z-axis is already aligned downward.")
        return T_b_approach  # No adjustment needed

    # Compute the axis and angle for rotation
    rotation_axis = np.cross(current_z, target_z)
    angle = np.arccos(np.dot(current_z, target_z) / (np.linalg.norm(current_z) * np.linalg.norm(target_z)))

    # Special case: If already aligned (no cross product possible)
    if np.linalg.norm(rotation_axis) < 1e-6:
        print("No rotation needed as the current_z is opposite to target_z.")
        return T_b_approach

    # Compute the rotation matrix
    rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, angle)

    # Apply the rotation
    T_b_approach[:3, :3] = rotation_matrix @ original_rotation
    print("Z-axis aligned to downward (-z).")
    return T_b_approach

def move_to_block(block_pose, current_joint_angles, arm, ik_solver, fk_solver, T_ee_camera):
    """
    Moves the robot end effector to a block's position while ensuring it always faces downward along the z-axis
    and aligns correctly for grasping the block.

    Parameters:
        block_pose (np.array): 4x4 transformation matrix of the block in the camera frame.
        current_joint_angles (np.array): Current joint angles of the robot.
        arm (ArmController): Instance of the ArmController.
        ik_solver (IK): Inverse kinematics solver instance.
        fk_solver (FK): Forward kinematics solver instance.
        T_ee_camera (np.array): Transformation from end effector to camera frame.

    Returns:
        bool: True if the movement was successful, False otherwise.
    """
    # Get the base-to-end-effector transform
    _, T_base_ee = fk_solver.forward(current_joint_angles)

    # Compute the transform from base to block
    T_base_block = np.matmul(T_base_ee, np.matmul(T_ee_camera, block_pose))

    # Define a grasp approach position (slightly above the block)
    T_b_approach = deepcopy(T_base_block)
    T_b_approach[2, 3] += 0.1  # Approach from 10 cm above

    T_b_approach = adjust_rotation_for_downward_z(T_b_approach)

    # Solve IK to move to the block
    q_approach, _, success_approach, _ = ik_solver.inverse(T_b_approach, current_joint_angles, method="J_pseudo", alpha=1)

    if success_approach:
        print(f"IK Solution found for block at:\n{block_pose}. Moving...")
        arm.safe_move_to_position(q_approach)
        print(f"Reached approach position.")
        return True
    else:
        print(f"Failed to find IK solution for block at:\n{block_pose}.")
        return False


def pick_up_block(arm, ik_solver, fk_solver, drop_height=0.12):
    """
    Performs a pick-up operation by moving down 10 cm, closing the gripper, and moving back up.

    Parameters:
        arm (ArmController): Instance of the ArmController.
        ik_solver (IK): Inverse kinematics solver instance.
        fk_solver (FK): Forward kinematics solver instance.
        drop_height (float): Distance to drop down for grasping in meters.

    Returns:
        bool: True if the pick-up operation was successful, False otherwise.
    """
    # Get the current joint angles
    current_joint_angles = deepcopy(arm.get_positions())
    print(current_joint_angles)

    # Use forward kinematics to get the current end-effector position
    _, T_b_approach = fk_solver.forward(current_joint_angles)

    # Move down by drop_height
    T_b_grasp = deepcopy(T_b_approach)
    T_b_grasp[2, 3] -= drop_height  # Drop down in the z-direction

    # Solve IK for the drop-down position
    q_grasp, _, success_grasp, _ = ik_solver.inverse(T_b_grasp, arm.get_positions(), method="J_pseudo", alpha=1)

    if not success_grasp:
        print("Failed to find IK solution for grasp position.")
        return False

    # Move to the grasp position
    print("Moving to grasp position...")
    # arm.open_gripper()
    arm.safe_move_to_position(q_grasp)
    # arm.open_gripper()

    # Close the gripper to grasp the block
    print("Closing gripper...")
    arm.exec_gripper_cmd(pos=0.048, force=50)  # Adjust grip width and force as needed

    # Move back up to the original approach position
    print("Moving back to approach position...")
    arm.safe_move_to_position(current_joint_angles)  # Back to approach position

    print("Pick-up operation completed successfully.")
    print(current_joint_angles)
    return True


def place_block(arm, ik_solver, block_name, base_placement_pos, base_z, force=50):
    """
    Place a block at a specified location with stacking.

    Parameters:
        arm (ArmController): Robot arm controller
        ik_solver (IK): Inverse kinematics solver
        block_name (str): Name of the block being placed
        base_placement_pos (np.array): Base x, y coordinates for placement
        base_z (float): Base z coordinate for placement
        force (int): Gripper force for releasing block

    Returns:
        bool: True if block placement was successful, False otherwise
    """
    # Prepare target placement position with current z height
    target_placing = np.array([base_placement_pos[0], base_placement_pos[1], base_z])

    # Create transformation matrix for placing
    T_target = transform(target_placing, np.array([0, pi, pi]))

    # Get current arm position as reference
    current_position = arm.get_positions()

    # Solve IK for the drop position
    q_drop, _, success_drop, _ = ik_solver.inverse(T_target, current_position, method="J_pseudo", alpha=1)

    if success_drop:
        print(f"Found dropping action for {block_name}")

        # Move to drop position
        arm.safe_move_to_position(q_drop)

        # Open gripper to release block
        arm.exec_gripper_cmd(pos=0.06, force=force)

        return True
    else:
        print(f"Could not find dropping position for {block_name}")
        return False


def red_static_stack(static_scan_pos, arm, detector, ik_solver, fk_solver, T_ee_camera):
    """
    Executes the static block stacking procedure.

    Parameters:
        team (str): The team's color ('red' or 'blue').
        arm (ArmController): Instance of the ArmController.
        detector (ObjectDetector): Instance of the ObjectDetector.
        ik_solver (IK): Inverse kinematics solver instance.
        fk_solver (FK): Forward kinematics solver instance.
        T_ee_camera (np.array): Transformation from end effector to camera frame.

    Returns:
        None
        :param team:
        :param T_ee_camera:
        :param fk_solver:
        :param ik_solver:
        :param detector:
        :param arm:
        :param static_scan_pos:
    """

    red_plate = [0.20257,  0.1237,   0.09365, -1.47087, -0.01154,  1.59404,  1.08118]

    arm.safe_move_to_position(static_scan_pos)

    # Detect and filter static blocks
    detections = detector.get_detections()
    static_blocks = [detection for detection in detections if "_static" in detection[0]]

    # Get the current joint angles
    current_joint_angles = arm.get_positions()

    try:
        # Move to the first detected block
        first_block_name, first_block_pose = static_blocks[0]
        print(f"Moving to first block: {first_block_name} at pose:\n{first_block_pose}")

        success = move_to_block(
            block_pose=np.array(first_block_pose),
            current_joint_angles=current_joint_angles,
            arm=arm,
            ik_solver=ik_solver,
            fk_solver=fk_solver,
            T_ee_camera=T_ee_camera
        )

        if success:
            # Open gripper
            arm.open_gripper()

            # Pick up block
            print(f"Picking up block: {first_block_name}")
            success_pickup = pick_up_block(arm, ik_solver, fk_solver)

            if success_pickup:
                arm.safe_move_to_position(red_plate)
                q1 = [0.27527,  0.26074,  0.01355, -2.0181,  -0.0046,   2.27881,  1.07675]
                arm.safe_move_to_position(q1)
                # Open gripper
                arm.open_gripper()
                arm.safe_move_to_position(red_plate)

            else:
                print("Failed to pick up block")
    except IndexError:
        print("Block 1 not detected")
        return

##########################BLOCK 2 ##############################################
    try:
        arm.safe_move_to_position(static_scan_pos)

        # Get the current joint angles
        current_joint_angles = arm.get_positions()

        # Move to the first detected block
        second_block_name, second_block_pose = static_blocks[1]
        print(f"Moving to second block: {second_block_name} at pose:\n{second_block_pose}")

        success = move_to_block(
            block_pose=np.array(second_block_pose),
            current_joint_angles=current_joint_angles,
            arm=arm,
            ik_solver=ik_solver,
            fk_solver=fk_solver,
            T_ee_camera=T_ee_camera
        )

        if success:
            # Open gripper
            arm.open_gripper()

            # Pick up block
            print(f"Picking up block: {second_block_name}")
            success_pickup = pick_up_block(arm, ik_solver, fk_solver)

            if success_pickup:
                arm.safe_move_to_position(red_plate)
                q2 = [0.24003,  0.19568,  0.05061, -1.97048, -0.01188,  2.16588,  1.08174]
                arm.safe_move_to_position(q2)
                # Open gripper
                arm.open_gripper()
                arm.safe_move_to_position(red_plate)

            else:
                print("Failed to pick up block")
    except IndexError:
        print("Block 2 not detected")
        return

##########################BLOCK 3 ##############################################

    try:
        arm.safe_move_to_position(static_scan_pos)

        # Get the current joint angles
        current_joint_angles = arm.get_positions()

        # Move to the first detected block
        third_block_name, third_block_pose = static_blocks[2]
        print(f"Moving to third block: {third_block_name} at pose:\n{third_block_pose}")

        success = move_to_block(
            block_pose=np.array(third_block_pose),
            current_joint_angles=current_joint_angles,
            arm=arm,
            ik_solver=ik_solver,
            fk_solver=fk_solver,
            T_ee_camera=T_ee_camera
        )

        if success:
            # Open gripper
            arm.open_gripper()

            # Pick up block
            print(f"Picking up block: {second_block_name}")
            success_pickup = pick_up_block(arm, ik_solver, fk_solver)

            if success_pickup:
                arm.safe_move_to_position(red_plate)
                q3 = [0.21444,  0.14668,  0.0775,  -1.90308, -0.01275,  2.0493,   1.08238]
                arm.safe_move_to_position(q3)
                # Open gripper
                arm.open_gripper()
                arm.safe_move_to_position(red_plate)

            else:
                print("Failed to pick up block")
    except IndexError:
        print("Block 3 not detected")
        return

##########################BLOCK 4 ##############################################

    try:
        arm.safe_move_to_position(static_scan_pos)

        # Get the current joint angles
        current_joint_angles = arm.get_positions()

        # Move to the first detected block
        fourth_block_name, fourth_block_pose = static_blocks[3]
        print(f"Moving to fourth block: {fourth_block_name} at pose:\n{fourth_block_pose}")

        success = move_to_block(
            block_pose=np.array(fourth_block_pose),
            current_joint_angles=current_joint_angles,
            arm=arm,
            ik_solver=ik_solver,
            fk_solver=fk_solver,
            T_ee_camera=T_ee_camera
        )

        if success:
            # Open gripper
            arm.open_gripper()

            # Pick up block
            print(f"Picking up block: {fourth_block_name}")
            success_pickup = pick_up_block(arm, ik_solver, fk_solver)

            if success_pickup:
                arm.safe_move_to_position(red_plate)
                q4 = [0.19836,  0.11478,  0.09437, -1.81512, -0.01153,  1.92937,  1.08156]
                arm.safe_move_to_position(q4)
                # Open gripper
                arm.open_gripper()
                arm.safe_move_to_position(red_plate)

            else:
                print("Failed to pick up block")
    except IndexError:
        print("Block 4 not detected")
        return


def blue_static_stack(static_scan_pos, arm, detector, ik_solver, fk_solver, T_ee_camera):
    """
    Executes the static block stacking procedure.

    Parameters:
        team (str): The team's color ('red' or 'blue').
        arm (ArmController): Instance of the ArmController.
        detector (ObjectDetector): Instance of the ObjectDetector.
        ik_solver (IK): Inverse kinematics solver instance.
        fk_solver (FK): Forward kinematics solver instance.
        T_ee_camera (np.array): Transformation from end effector to camera frame.

    Returns:
        None
        :param team:
        :param T_ee_camera:
        :param fk_solver:
        :param ik_solver:
        :param detector:
        :param arm:
        :param static_scan_pos:
    """

    blue_plate = [-0.17425,  0.28042, -0.15679, -0.98256,  0.0454,   1.26001,  0.47429]

    arm.safe_move_to_position(static_scan_pos)

    # Detect and filter static blocks
    detections = detector.get_detections()
    static_blocks = [detection for detection in detections if "_static" in detection[0]]

    # Get the current joint angles
    current_joint_angles = arm.get_positions()

    try:
        # Move to the first detected block
        first_block_name, first_block_pose = static_blocks[0]
        print(f"Moving to first block: {first_block_name} at pose:\n{first_block_pose}")

        success = move_to_block(
            block_pose=np.array(first_block_pose),
            current_joint_angles=current_joint_angles,
            arm=arm,
            ik_solver=ik_solver,
            fk_solver=fk_solver,
            T_ee_camera=T_ee_camera
        )

        if success:
            # Open gripper
            arm.open_gripper()

            # Pick up block
            print(f"Picking up block: {first_block_name}")
            success_pickup = pick_up_block(arm, ik_solver, fk_solver)

            if success_pickup:
                arm.safe_move_to_position(blue_plate)
                q1 = [-0.11187,  0.26504, -0.18444, -2.01773,  0.06321,  2.27749,  0.45432]
                arm.safe_move_to_position(q1)
                # Open gripper
                arm.open_gripper()
                arm.safe_move_to_position(blue_plate)

            else:
                print("Failed to pick up block")
    except IndexError:
        print("Block 1 not detected")
        return

    ##########################BLOCK 2 ##############################################

    try:
        arm.safe_move_to_position(static_scan_pos)

        # Get the current joint angles
        current_joint_angles = arm.get_positions()

        # Move to the first detected block
        second_block_name, second_block_pose = static_blocks[1]
        print(f"Moving to second block: {second_block_name} at pose:\n{second_block_pose}")

        success = move_to_block(
            block_pose=np.array(second_block_pose),
            current_joint_angles=current_joint_angles,
            arm=arm,
            ik_solver=ik_solver,
            fk_solver=fk_solver,
            T_ee_camera=T_ee_camera
        )

        if success:
            # Open gripper
            arm.open_gripper()

            # Pick up block
            print(f"Picking up block: {second_block_name}")
            success_pickup = pick_up_block(arm, ik_solver, fk_solver)

            if success_pickup:
                arm.safe_move_to_position(blue_plate)
                q2 = [-0.12088,  0.19836, -0.1758,  -1.97033,  0.04162,  2.16531,  0.46877]
                arm.safe_move_to_position(q2)
                # Open gripper
                arm.open_gripper()
                arm.safe_move_to_position(blue_plate)

            else:
                print("Failed to pick up block")
    except IndexError:
        print("Block 2 not detected")
        return

    ##########################BLOCK 3 ##############################################

    try:
        arm.safe_move_to_position(static_scan_pos)

        # Get the current joint angles
        current_joint_angles = arm.get_positions()

        # Move to the first detected block
        third_block_name, third_block_pose = static_blocks[2]
        print(f"Moving to third block: {third_block_name} at pose:\n{third_block_pose}")

        success = move_to_block(
            block_pose=np.array(third_block_pose),
            current_joint_angles=current_joint_angles,
            arm=arm,
            ik_solver=ik_solver,
            fk_solver=fk_solver,
            T_ee_camera=T_ee_camera
        )

        if success:
            # Open gripper
            arm.open_gripper()

            # Pick up block
            print(f"Picking up block: {second_block_name}")
            success_pickup = pick_up_block(arm, ik_solver, fk_solver)

            if success_pickup:
                arm.safe_move_to_position(blue_plate)
                q3 = [-0.12816,  0.14826, -0.16818, -1.90302,  0.02786,  2.04906,  0.47804]
                arm.safe_move_to_position(q3)
                # Open gripper
                arm.open_gripper()
                arm.safe_move_to_position(blue_plate)

            else:
                print("Failed to pick up block")
    except IndexError:
        print("Block 3 not detected")
        return

    ##########################BLOCK 4 ##############################################

    try:
        arm.safe_move_to_position(static_scan_pos)

        # Get the current joint angles
        current_joint_angles = arm.get_positions()

        # Move to the first detected block
        fourth_block_name, fourth_block_pose = static_blocks[3]
        print(f"Moving to fourth block: {fourth_block_name} at pose:\n{fourth_block_pose}")

        success = move_to_block(
            block_pose=np.array(fourth_block_pose),
            current_joint_angles=current_joint_angles,
            arm=arm,
            ik_solver=ik_solver,
            fk_solver=fk_solver,
            T_ee_camera=T_ee_camera
        )

        if success:
            # Open gripper
            arm.open_gripper()

            # Pick up block
            print(f"Picking up block: {fourth_block_name}")
            success_pickup = pick_up_block(arm, ik_solver, fk_solver)

            if success_pickup:
                arm.safe_move_to_position(blue_plate)
                q4 = [-0.13328,  0.11575, -0.16276, -1.81509, 0.01999,  1.92927,  0.48341]
                arm.safe_move_to_position(q4)
                # Open gripper
                arm.open_gripper()
                arm.safe_move_to_position(blue_plate)

            else:
                print("Failed to pick up block")
    except IndexError:
        print("Block 4 not detected")
        return







####### DYNAMIC GRASPING #########

def move_to_dynamic_prepare(T_base_target, current_joint_angles, arm, ik_solver):
    q_dynamic_prepare, _, success_approach, _ = ik_solver.inverse(T_base_target, current_joint_angles, method="J_pseudo", alpha=0.1)

    if success_approach:
        arm.safe_move_to_position(q_dynamic_prepare)
        print("Reached dynamic prepare position successfully.")
        print("q_dynamic_prepare:")
        print(q_dynamic_prepare)
        return True
    else:
        print("Failed to find IK solution for dynamic prepare position.")
        return False


def grasp_and_check(arm):
    """
    Close the gripper and check if a block is grasped
    """
    arm.close_gripper()

    time.sleep(1)

    gripper_state = arm.get_gripper_state()
    if gripper_state['position'][0] + gripper_state['position'][1] > 0.03:
        print("Block grasped successfully.")
        return True
    else:
        print("Block not grasped.")
        print("gripper_state:")
        print(gripper_state)
        return False

def detection_in_world(detections, T_ee_camera, current_joint_angles, fk_solver):
    """
    Converts the detection array from camera coordinates to world coordinates.

    Parameters:
        detection (np.array): Detection array in camera coordinates.
        T_ee_camera (np.array): Transformation matrix from end effector to camera frame.
        T_ee_base (np.array): Transformation matrix from end effector to base frame.

    Returns:
        world_detections (list): List of tuples containing detection name and pose in world coordinates.
        world_detections_xyz (np.array): Array of detection positions in world coordinates.
        world_detections_R (np.array): Array of detection rotations in world coordinates.
    """
    # Get the base-to-end-effector transform
    _, T_base_ee = fk_solver.forward(current_joint_angles)

    # Transform each detection to world coordinates
    world_detections = []
    world_detections_trans = []
    world_rotations = []
    for name, pose in detections:
        T_world = np.matmul(T_base_ee, np.matmul(T_ee_camera, pose))
        world_detections.append((name+"_world", T_world))
        world_detections_trans.append((T_world[:3, 3]).tolist())
        world_rotations.append(T_world[:3, :3].tolist())

    return world_detections, np.array(world_detections_trans), np.array(world_rotations)

def find_intersection(circle_center, radius, line_slope, line_intercept):
    h, k = circle_center
    r = radius
    m = line_slope
    c = line_intercept

    # Coefficients for the quadratic equation ax^2 + bx + c = 0
    a = 1 + m**2
    b = 2 * m * (c - k) - 2 * h
    c = h**2 + (c - k)**2 - r**2

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return "No intersection points"
    elif discriminant == 0:
        # One intersection point (tangent)
        x = -b / (2 * a)
        y = m * x + line_intercept
        return [(x, y)]
    else:
        # Two intersection points
        sqrt_discriminant = np.sqrt(discriminant)
        x1 = (-b + sqrt_discriminant) / (2 * a)
        y1 = m * x1 + line_intercept
        x2 = (-b - sqrt_discriminant) / (2 * a)
        y2 = m * x2 + line_intercept
        return [(x1, y1), (x2, y2)]

def test_offset_red(arm, ik_solver, fk_solver):
    q_prepare = [-0.08071935, -1.05097604,  1.79328915, -2.1446664,   2.7348064,   2.35238015, -1.14184532]
    q_grasp = [-0.50350687, -1.09917709, 1.95515374, -1.70513756, 2.5476664, 2.24006833, -1.0316845]
    q_lift = [-0.45400673, -0.96767907, 1.85857134, -1.69441744, 2.34406776, 2.27635553, -0.89763929]

    q_test = q_prepare

    arm.open_gripper()

    input("Press Enter to go to test position...")
    arm.safe_move_to_position(q_test)

    _, T_base_ee = fk_solver.forward(q_test)

    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] -= 0.085
    print("T_base_ee:", T_base_ee[:3, 3])
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

def test_offset_blue(arm, ik_solver, fk_solver):
    q_prepare = [2.4154031, -0.42689215,  2.22888276, -2.1815192,   2.29061197,  2.24240731, -1.50512913]
    q_grasp = [1.91427539, -0.71243673,  2.54415673, -1.65009661,  1.85481512,  1.89600413, -1.30513994]
    q_lift = [2.04095641, -0.49232606,  2.36823143, -1.55741683,  1.6625471,  1.88678225, -0.97267072]

    q_test = q_prepare

    arm.open_gripper()

    input("Press Enter to go to test position...")
    arm.safe_move_to_position(q_test)

    _, T_base_ee = fk_solver.forward(q_test)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)


def dynamic_grasping_red(arm, ik_solver, fk_solver, detector, T_ee_camera):
    # q_prepare = [-0.08071935, -1.05097604,  1.79328915, -2.1446664,   2.7348064,   2.35238015, -1.14184532]
    q_prepare_real = [-0.52017087, -1.70708483,  1.77208219, -0.96912002,  2.23791465,  2.90416482, 0.1054177]
    # q_grasp = [-0.50350687, -1.09917709, 1.95515374, -1.70513756, 2.5476664, 2.24006833, -1.0316845]
    q_grasp_real = [-0.49732725, -1.56347996, 1.83934845, -1.33414957, 2.64071319, 2.66219545, -0.48008836]
    q_lift = [-0.45400673, -0.96767907, 1.85857134, -1.69441744, 2.34406776, 2.27635553, -0.89763929]
    q_prebuild = [-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866]
    q_drop = [ 0.29184179, 0.00440474, 0.29431299, -2.02162632, 1.83832933, 3.6621717, -0.77207898]
    q_home = [-0.00001, -0.784994, 0.000026, -2.355987, 0.000008, 1.569987, 0.784997]

    arm.open_gripper()

    print("Moving to prepare position...")
    arm.safe_move_to_position(q_prepare_real)

    # print("Rotating 10 degrees around x...")

    # _, T_base_ee = fk_solver.forward(q_prepare)
    # R_n10_x = rotation_x(-10)
    # T_base_ee[:3, :3] = T_base_ee[:3, :3] @ R_n10_x
    # success = move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)


    input("Press Enter to go to grasp position...")
    arm.safe_move_to_position(q_grasp_real)

    _, T_base_ee = fk_solver.forward(q_grasp_real)

    grasp_success = False
    threshold = 0.93 # TODO: adjust this

    while not grasp_success:
        arm.open_gripper()
        current_joint_angles = arm.get_positions()
        detections = detector.get_detections()
        _, blocks_pos, _ = detection_in_world(detections, T_ee_camera, current_joint_angles, fk_solver)

        # if the smallest y < threshold, close gripper
        min_y = min(blocks_pos[:, 1])
        print("min_y:", min_y)
        if min_y < threshold:
            print("Waiting...")
            time.sleep(2.0) # TODO: adjust this
            while not grasp_success:
                print("Grasping...")
                grasp_success = grasp_and_check(arm)
                if not grasp_success:
                    arm.open_gripper()
                    time.sleep(1.0)


    # Brute force method
    # while not grasp_success:
    #     open_gripper(arm)
    #     time.sleep(1.0) # TODO: adjust this
    #     print("Grasping...")
    #     grasp_success = grasp_and_check(arm)


    print("Lifting up the block...")
    arm.safe_move_to_position(q_lift)

    print("Moving to pre-build position...")
    arm.safe_move_to_position(q_prebuild)

    print("Dropping the block...")
    arm.safe_move_to_position(q_drop)

    arm.open_gripper()

    print("Moving to home position...")
    arm.safe_move_to_position(q_home)



def dynamic_grasping_blue(arm, ik_solver, fk_solver, detector, T_ee_camera):
    # q_prepare = [2.4154031, -0.42689215,  2.22888276, -2.1815192,   2.29061197,  2.24240731, -1.50512913]
    q_prepare_real = [2.62, -1.70708483,  1.77208219, -0.96912002,  2.23791465,  2.90416482, 0.1054177]
    # q_grasp = [1.91427539, -0.71243673,  2.54415673, -1.65009661,  1.85481512,  1.89600413, -1.30513994]
    q_grasp_real = [ 2.59078109, -1.43426034,  1.98162611, -1.28371041,  2.60510191,  2.74917688, -0.62866409]
    q_lift = [2.04095641, -0.49232606,  2.36823143, -1.55741683,  1.6625471,  1.88678225, -0.97267072]
    q_prebuild = [-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866]
    q_drop = [0.90971565, -0.414771, -1.10812814, -2.45677545, 0.3267401, 3.29784387, 0.06385336]
    q_raise = [-0.0351134, -0.2881468, -0.19028124, -2.25704634, 0.8628828, 2.92949924, -0.22332445]
    q_home = [-0.00001, -0.784994, 0.000026, -2.355987, 0.000008, 1.569987, 0.784997]

    ignore_distance = -0.90   # ignore blocks with distance below ignore_distance
    threshold_offset = 0.01 # distance of the parallel line from the gripper's tip

    arm.open_gripper()

    # input("Press Enter to go to prepare position...")

    print("Moving to prepare position...")
    arm.safe_move_to_position(q_prepare_real)

    # get radius of the target block's circular path
    blocks_pos_filtered = []
    while len(blocks_pos_filtered) == 0:
        current_joint_angles = arm.get_positions()
        detections = detector.get_detections()
        _, blocks_pos, _ = detection_in_world(detections, T_ee_camera, current_joint_angles, fk_solver)
        blocks_pos_filtered = blocks_pos[blocks_pos[:, 1] < ignore_distance]    # TODO: change to > for red
        print("blocks_pos_filtered:", blocks_pos_filtered)
        if len(blocks_pos_filtered) > 0:
            closest_block_pos = blocks_pos_filtered[np.argmin(blocks_pos_filtered[:, 1])]
            block_pos_2d = closest_block_pos[:2]
            disc_center = np.array([0.0, -0.99])
            r = np.linalg.norm(block_pos_2d - disc_center)
            print("r:", r)
        else:
            print("No blocks found within the ignore distance.")

    # input("Press Enter to go to grasp position...")
    arm.safe_move_to_position(q_grasp_real)

    _, T_base_ee = fk_solver.forward(q_grasp_real)

    T_ee_p1 = np.array([0, 0.04, threshold_offset, 1]).T
    T_ee_p2 = np.array([0, -0.04, threshold_offset, 1]).T

    T_base_p1 = T_base_ee @ T_ee_p1
    T_base_p2 = T_base_ee @ T_ee_p2

    # use the projection of the points onto the xy plane
    P1 = T_base_p1[:2]   # P1 in base frame
    P2 = T_base_p2[:2]   # P2 in base frame

    vec_AB = P2 - P1
    m = vec_AB[1] / vec_AB[0]
    c = P1[1] - m * P1[0]

    # find the intersection of line AB with the block's circular path
    intersections = find_intersection(disc_center, r, m, c)
    print("intersections:", intersections)

    # keep intersection with the larger y value (for blue)
    # TODO: change for red
    if intersections[0][1] < intersections[1][1]:
        intersection = intersections[1]
    else:
        intersection = intersections[0]

    # get intersection in EE frame (P0i = T0e @ Pei)
    P_ee_intersection = np.linalg.inv(T_base_ee) @ np.array([intersection[0], intersection[1], 0.0, 1.0]).T
    print("P_ee_intersection:", P_ee_intersection)
    gripper_offset = P_ee_intersection[1]    # offset in y direction
    print("gripper_offset:", gripper_offset)

    T_base_ee[1, 3] += gripper_offset
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)
    print("Offset added")

    grasp_success = False

    while not grasp_success:
        arm.open_gripper()
        current_joint_angles = arm.get_positions()
        detections = detector.get_detections()
        _, blocks_pos, _ = detection_in_world(detections, T_ee_camera, current_joint_angles, fk_solver)

        block_in_range = False
        for i in range(len(blocks_pos)):
            vec_AC = blocks_pos[i][:2] - P1
            cross_product = np.cross(vec_AB, vec_AC)
            print("cross_product for block", i, ":", cross_product)
            if cross_product > 0:
                block_in_range = True
                break

        print("")

        if block_in_range:
            # time.sleep(2.0) # TODO: adjust this
            while not grasp_success:
                print("Grasping...")
                grasp_success = grasp_and_check(arm)
                if not grasp_success:
                    arm.open_gripper()
                    time.sleep(1.0)


    # Brute force method
    while not grasp_success:
        arm.open_gripper()
        time.sleep(10.0) # TODO: adjust this
        print("Grasping...")
        grasp_success = grasp_and_check(arm)


    print("Lifting up the block...")
    arm.safe_move_to_position(q_lift)

    print("Moving to pre-build position...")
    arm.safe_move_to_position(q_prebuild)

    print("Lowering the block...")
    arm.safe_move_to_position(q_drop)

    arm.open_gripper()

    print("Raising the gripper...")
    arm.safe_move_to_position(q_raise)

    print("Moving to home position...")
    arm.safe_move_to_position(q_home)







def dynamic_grasping_red_brute(arm, ik_solver, fk_solver, detector, T_ee_camera):
    # q_prepare = [-0.08071935, -1.05097604,  1.79328915, -2.1446664,   2.7348064,   2.35238015, -1.14184532]
    q_prepare_real = [-0.52017087, -1.70708483,  1.77208219, -0.96912002,  2.23791465,  2.90416482, 0.1054177]
    # q_grasp = [-0.50350687, -1.09917709, 1.95515374, -1.70513756, 2.5476664, 2.24006833, -1.0316845]
    q_grasp_real = [-0.49732725, -1.56347996, 1.83934845, -1.33414957, 2.64071319, 2.66219545, -0.48008836]
    q_lift = [-0.45400673, -0.96767907, 1.85857134, -1.69441744, 2.34406776, 2.27635553, -0.89763929]
    q_prebuild = [-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866]
    q_drop = [ 0.29184179, 0.00440474, 0.29431299, -2.02162632, 1.83832933, 3.6621717, -0.77207898]
    q_raise = [ 0.29184544, 0.00440638, 0.29430607, -1.84711479, 1.83830974, 3.66216323, -0.77204947]
    q_home = [-0.00001, -0.784994, 0.000026, -2.355987, 0.000008, 1.569987, 0.784997]

    arm.open_gripper()

    print("Moving to prepare position...")
    # arm.safe_move_to_position(q_prepare_real)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_prepare_real)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    print("Moving to grasp position...")
    # arm.safe_move_to_position(q_grasp_real)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_grasp_real)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    grasp_success = False

    # Brute force method
    while not grasp_success:
        arm.open_gripper()
        time.sleep(10.0) # TODO: adjust this
        print("Grasping...")
        grasp_success = grasp_and_check(arm)


    print("Lifting up the block...")
    # arm.safe_move_to_position(q_lift)
    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_lift)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    print("Moving to pre-build position...")
    arm.safe_move_to_position(q_prebuild)

    print("Dropping the block...")
    # arm.safe_move_to_position(q_drop)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_drop)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    arm.open_gripper()

    print("Raising the gripper...")
    arm.safe_move_to_position(q_raise)


    # BLOCK 2
    arm.open_gripper()

    print("Moving to prepare position...")
    # arm.safe_move_to_position(q_prepare_real)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_prepare_real)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    print("Moving to grasp position...")
    # arm.safe_move_to_position(q_grasp_real)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_grasp_real)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    grasp_success = False

    # Brute force method
    while not grasp_success:
        arm.open_gripper()
        time.sleep(10.0) # TODO: adjust this
        print("Grasping...")
        grasp_success = grasp_and_check(arm)


    print("Lifting up the block...")
    # arm.safe_move_to_position(q_lift)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_lift)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    print("Moving to pre-build position...")
    arm.safe_move_to_position(q_prebuild)

    print("Dropping the block...")
    # arm.safe_move_to_position(q_drop)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_drop)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0.05
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    arm.open_gripper()

    print("Raising the gripper...")
    arm.safe_move_to_position(q_raise)






def dynamic_grasping_blue_brute(arm, ik_solver, fk_solver, detector, T_ee_camera):
    # q_prepare = [2.4154031, -0.42689215,  2.22888276, -2.1815192,   2.29061197,  2.24240731, -1.50512913]
    q_prepare_real = [2.62, -1.70708483,  1.77208219, -0.96912002,  2.23791465,  2.90416482, 0.1054177]
    # q_grasp = [1.91427539, -0.71243673,  2.54415673, -1.65009661,  1.85481512,  1.89600413, -1.30513994]
    q_grasp_real = [ 2.59078109, -1.43426034,  1.98162611, -1.28371041,  2.60510191,  2.74917688, -0.62866409]
    q_lift = [2.04095641, -0.49232606,  2.36823143, -1.55741683,  1.6625471,  1.88678225, -0.97267072]
    q_prebuild = [-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866]
    q_drop = [0.90971565, -0.414771, -1.10812814, -2.45677545, 0.3267401, 3.29784387, 0.06385336]
    q_raise = [-0.0351134, -0.2881468, -0.19028124, -2.25704634, 0.8628828, 2.92949924, -0.22332445]
    q_home = [-0.00001, -0.784994, 0.000026, -2.355987, 0.000008, 1.569987, 0.784997]

    arm.open_gripper()

    # input("Press Enter to go to prepare position...")

    print("Moving to prepare position...")
    # arm.safe_move_to_position(q_prepare_real)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_prepare_real)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    print("Moving to grasp position...")
    # arm.safe_move_to_position(q_grasp_real)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_grasp_real)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    grasp_success = False

    # Brute force method
    while not grasp_success:
        arm.open_gripper()
        time.sleep(10.0) # TODO: adjust this
        print("Grasping...")
        grasp_success = grasp_and_check(arm)


    print("Lifting up the block...")
    # arm.safe_move_to_position(q_lift)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_lift)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    print("Moving to pre-build position...")
    arm.safe_move_to_position(q_prebuild)

    print("Lowering the block...")
    # arm.safe_move_to_position(q_drop)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_drop)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    arm.open_gripper()

    print("Raising the gripper...")
    arm.safe_move_to_position(q_raise)



    # BLOCK 2
    arm.open_gripper()

    # input("Press Enter to go to prepare position...")

    print("Moving to prepare position...")
    # arm.safe_move_to_position(q_prepare_real)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_prepare_real)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    print("Moving to grasp position...")
    # arm.safe_move_to_position(q_grasp_real)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_grasp_real)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    grasp_success = False

    # Brute force method
    while not grasp_success:
        arm.open_gripper()
        time.sleep(10.0) # TODO: adjust this
        print("Grasping...")
        grasp_success = grasp_and_check(arm)


    print("Lifting up the block...")
    # arm.safe_move_to_position(q_lift)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_lift)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    print("Moving to pre-build position...")
    arm.safe_move_to_position(q_prebuild)

    print("Lowering the block...")
    # arm.safe_move_to_position(q_drop)

    # ADD OFFSET IF NECESSARY
    _, T_base_ee = fk_solver.forward(q_drop)
    T_base_ee[0, 3] += 0
    T_base_ee[1, 3] += 0
    T_base_ee[2, 3] += 0.05
    move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)

    arm.open_gripper()

    print("Raising the gripper...")
    arm.safe_move_to_position(q_raise)






def red_test_offset(arm, ik_solver, fk_solver):
    # q_prepare = [-0.08071935, -1.05097604,  1.79328915, -2.1446664,   2.7348064,   2.35238015, -1.14184532]
    q_prepare_real = [-0.52017087, -1.70708483,  1.77208219, -0.96912002,  2.23791465,  2.90416482, 0.1054177]
    # q_grasp = [-0.50350687, -1.09917709, 1.95515374, -1.70513756, 2.5476664, 2.24006833, -1.0316845]
    q_grasp_real = [-0.49732725, -1.56347996, 1.83934845, -1.33414957, 2.64071319, 2.66219545, -0.48008836]
    q_lift = [-0.45400673, -0.96767907, 1.85857134, -1.69441744, 2.34406776, 2.27635553, -0.89763929]
    q_prebuild = [-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866]
    q_drop = [ 0.29184179, 0.00440474, 0.29431299, -2.02162632, 1.83832933, 3.6621717, -0.77207898]
    q_raise = [ 0.29184544, 0.00440638, 0.29430607, -1.84711479, 1.83830974, 3.66216323, -0.77204947]
    q_home = [-0.00001, -0.784994, 0.000026, -2.355987, 0.000008, 1.569987, 0.784997]

    arm.safe_move_to_position(q_drop)

    # _, T_base_ee = fk_solver.forward(q_drop)
    # T_base_ee[0, 3] += 0.0
    # T_base_ee[1, 3] += 0.0
    # T_base_ee[2, 3] += 0.0
    # move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)


def blue_test_offset(arm, ik_solver, fk_solver):
    # q_prepare = [2.4154031, -0.42689215,  2.22888276, -2.1815192,   2.29061197,  2.24240731, -1.50512913]
    q_prepare_real = [2.62, -1.70708483,  1.77208219, -0.96912002,  2.23791465,  2.90416482, 0.1054177]
    # q_grasp = [1.91427539, -0.71243673,  2.54415673, -1.65009661,  1.85481512,  1.89600413, -1.30513994]
    q_grasp_real = [ 2.59078109, -1.43426034,  1.98162611, -1.28371041,  2.60510191,  2.74917688, -0.62866409]
    q_lift = [2.04095641, -0.49232606,  2.36823143, -1.55741683,  1.6625471,  1.88678225, -0.97267072]
    q_prebuild = [-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866]
    q_drop = [0.90971565, -0.414771, -1.10812814, -2.45677545, 0.3267401, 3.29784387, 0.06385336]
    q_raise = [-0.0351134, -0.2881468, -0.19028124, -2.25704634, 0.8628828, 2.92949924, -0.22332445]
    q_home = [-0.00001, -0.784994, 0.000026, -2.355987, 0.000008, 1.569987, 0.784997]

    arm.safe_move_to_position(q_drop)

    # _, T_base_ee = fk_solver.forward(q_drop)
    # T_base_ee[0, 3] += 0.0
    # T_base_ee[1, 3] += 0.0
    # T_base_ee[2, 3] += 0.0
    # move_to_dynamic_prepare(T_base_ee, arm.get_positions(), arm, ik_solver)






def main():
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")

    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # Initialize solvers
    fk_solver = FK()
    ik_solver = IK()

    # Scanning position for static blocks
    static_scan_pos = arm.neutral_position()
    #static_scan_pos = static_scan_pos + np.array([0, 0, 0, 0.6, 0, 0, 0])
    if team == 'red':
        static_scan_pos = np.array([-0.1501,   0.02464, -0.15577, -1.59672,  0.00383,  1.62106,  0.47938])
    else:
        static_scan_pos = np.array([0.16391,  0.02459,  0.14172, -1.59672, -0.00348,  1.62106,  1.09117])
    #static_scan_pos = np.array([-0.17314, -0.10026, -0.14347, -1.74265, -0.01435,  1.64341,  0.47054])

    # Base placement position for stacking
    base_placement_pos = np.array([0.57, 0.169])

    # Starting z height for stacking
    base_z_height = 0.27

    # Get the transform from end-effector to camera
    T_ee_camera = detector.get_H_ee_camera()

    # Call the static stack function
    if team == 'red':
        red_static_stack(static_scan_pos, arm, detector, ik_solver, fk_solver, T_ee_camera)
    else:
        blue_static_stack(static_scan_pos, arm, detector, ik_solver, fk_solver, T_ee_camera)

    if team == 'red':
        dynamic_grasping_red_brute(arm, ik_solver, fk_solver, detector, T_ee_camera)
    else:
        dynamic_grasping_blue_brute(arm, ik_solver, fk_solver, detector, T_ee_camera)

    # if team == 'red':
    #     dynamic_grasping_red_brute(arm, ik_solver, fk_solver, detector, T_ee_camera)
    # else:
    #     dynamic_grasping_blue_brute(arm, ik_solver, fk_solver, detector, T_ee_camera)

    # if team == 'red':
    #     red_test_offset(arm, ik_solver, fk_solver)
    # else:
    #     blue_test_offset(arm, ik_solver, fk_solver)


if __name__ == "__main__":
    main()







    # # Detect some blocks...
    # for (name, pose) in detector.get_detections():
    #      print(name,'\n',pose)

    # Uncomment to get middle camera depth/rgb images
    # mid_depth = detector.get_mid_depth()
    # mid_rgb = detector.get_mid_rgb()

    # Move around...

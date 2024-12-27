import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy

from lib.calculateFK import FK


def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    fk = FK()

    # initialize path
    path = []
    t_start = [start]
    t_goal = [goal]
    parent_start = {tuple(start): None}
    parent_goal = {tuple(goal): None}
    path_found = False
    connect_pose = []
    max_iter = 1000

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    def is_joints_valid(joints, obstacles):
        for obstacle in obstacles: 
            if True in detectCollision(joints[:7], joints[1:], obstacle):
                return False
        return True

    # function to check collision through interpolated steps
    def is_collision_free(start_pose, end_pose, obstacles):

        step_size = 0.05
        diff = end_pose - start_pose
        step = np.linalg.norm(diff) / step_size

        for i in range(int(step)):
            prev_pose = start_pose + i * step_size * diff
            new_pose = start_pose + (i+1) * step_size * diff

            prev_joints, _ = fk.forward(prev_pose)
            new_joints, _ = fk.forward(new_pose)

            if not is_joints_valid(prev_joints, obstacles):
                return False
            if not is_joints_valid(new_joints, obstacles):
                return False

            for obstacle in obstacles:
                if True in detectCollision(prev_joints, new_joints, obstacle):
                    return False

        return True

    # check validity of start and goal poses
    start_joints, _ = fk.forward(start)
    if not is_joints_valid(start_joints, map.obstacles):
        print("Invalid start pose.")
        return np.array([])

    goal_joints, _ = fk.forward(goal)
    if not is_joints_valid(goal_joints, map.obstacles):
        print("Invalid goal pose.")
        return np.array([])

    # build the planning tree
    for iter in range(max_iter):

        start_found = False
        goal_found = False

        # randomly sample a pose
        q_new = np.random.uniform(lowerLim, upperLim)
        joint_positions, _ = fk.forward(q_new)

        # check if the sample has collision
        if not is_joints_valid(joint_positions, map.obstacles):
            continue

        # find nearest node in start tree and check collision
        start_distances = [np.linalg.norm(np.array(n) - q_new) for n in t_start]
        nearest_start_node = t_start[np.argmin(start_distances)]
        if is_collision_free(nearest_start_node, q_new, map.obstacles):
            t_start.append(q_new)
            parent_start[tuple(q_new)] = nearest_start_node
            start_found = True

        # find nearest node in goal tree and check collision
        goal_distances = [np.linalg.norm(np.array(n) - q_new) for n in t_goal]
        nearest_goal_node = t_goal[np.argmin(goal_distances)]
        if is_collision_free(nearest_goal_node, q_new, map.obstacles):
            t_goal.append(q_new)
            parent_goal[tuple(q_new)] = nearest_goal_node
            goal_found = True

        # found path if a pose both in start tree and goal tree
        if start_found & goal_found:
            path_found = True
            connect_pose = q_new
            break
    
    if not path_found:
        print("Path not found.")
        return np.array(path)

    # reconstruct the path
    node = connect_pose
    while node is not None:
        path.insert(0, node)
        node = parent_start.get(tuple(node))

    node = parent_goal.get(tuple(connect_pose))
    while node is not None:
        path.append(node)
        node = parent_goal.get(tuple(node))

    print("Path found!")
    return np.array(path)

if __name__ == '__main__':
    map_struct = loadmap("../maps/emptyMap.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

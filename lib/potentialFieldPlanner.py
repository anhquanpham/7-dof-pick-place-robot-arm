import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=0.02, max_steps =40000, min_step_size=1e-5): 
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current, joint_number):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """
	
       
        
        d = 0.12 
        
        if joint_number < 7:
            k_parabola = 30 
        else:
            k_parabola = 10 

        att_f = np.zeros((3, 1)) 
        
        diff = current - target ##3x1 numpy array
        
        dist_squared = np.linalg.norm(diff) #magnitude numerical value
        
        if dist_squared > d:
            # Use conic function:
            att_f = - diff / np.linalg.norm(diff) #3x1 array
        else:
            # Use parabolic function: 
            att_f = - k_parabola * diff  # 3x1 array

        
        
        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        
        rep_f = np.zeros((3, 1)) 
        k_rep = 0.003 #(repulsive field strength)
        rho_0 = 0.12 #distance of influence 
        
        distance, unitvec = PotentialFieldPlanner.dist_point2box(current.T, obstacle)
        #distance here is o - b
        unit_vector = unitvec.reshape(3,-1) #reshape unit vector from nx3 to 3xn
        
        if (distance > 0) & (distance <= rho_0):   
            rep_f = - k_rep * (1/distance - 1/rho_0) * (1/(distance**2)) * unit_vector
        else:   
            rep_f = np.zeros((3, 1))

        

        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        """
        #9 JOINTS HERE ARE 1 2 3 4 5 6 end virtual_1 virtual_2 

        

        joint_forces = np.zeros((3, 9)) 
        
        # Loop thru each to calculate the F_attractive force, then consider each obstacle to obtain its F_rep
        for i in range(9): #i runs from 0 to 8
            joint_number = i
            target_i = target[i].reshape(3, 1)
            current_i = current[i].reshape(3, 1)
            F_att = PotentialFieldPlanner.attractive_force(target_i, current_i, joint_number)
            F_net = F_att
            for j in range (len(obstacle)):
                F_rep = PotentialFieldPlanner.repulsive_force(obstacle[j,:], current_i)
                F_net = F_net + F_rep
            joint_forces[:, i] = F_net.reshape(3) #Reshape F_net from (3,1) to (3)

        

        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint 
        """

        

        # Get joint positions and transformations
        joints, T0e = PotentialFieldPlanner.fk.forward_expanded(q.reshape(7))
        
        # Extract each joint position as a vector
        o_0 = joints[0, :]  # First row (joint1)
        o_1 = joints[1, :]  # Second row (joint2)
        o_2 = joints[2, :]  # Third row (joint3)
        o_3 = joints[3, :]  # Fourth row (joint4)
        o_4 = joints[4, :]  # Fifth row (joint5)
        o_5 = joints[5, :]  # Sixth row (joint6)
        o_6 = joints[6, :]  # Seventh row (joint7)
        o_e = joints[7, :]  # Eighth row (end-effector)
        o_virtual_1 = joints[8, :] #1st virtual joint
        o_virtual_2 = joints[9, :] #2nd virtual joint
        
        z_0 = T0e[0][:3, 2] #Third column of transformation
        z_1 = T0e[1][:3, 2]
        z_2 = T0e[2][:3, 2]
        z_3 = T0e[3][:3, 2]
        z_4 = T0e[4][:3, 2]
        z_5 = T0e[5][:3, 2]
        z_6 = T0e[6][:3, 2]
        z_e = T0e[7][:3, 2]
        z_virtual_1 = T0e[8][:3, 2]
        z_virtual_2 = T0e[9][:3, 2]
        
        #Jv0
        Jv0 = np.zeros((3, 9))
        
        #Jv1
        Jv1 = np.zeros((3, 9))
        Jv1[:, 0] = np.cross(z_0, (o_1 - o_0))
        
        #Jv2
        Jv2 = np.zeros((3,9))
        Jv2[:, 0] = np.cross(z_0, (o_2 - o_0))
        Jv2[:, 1] = np.cross(z_1, (o_2 - o_1))
        
        #Jv3
        Jv3 = np.zeros((3,9))
        Jv3[:, 0] = np.cross(z_0, (o_3 - o_0))
        Jv3[:, 1] = np.cross(z_1, (o_3 - o_1))
        Jv3[:, 2] = np.cross(z_2, (o_3 - o_2))
        
        #Jv4
        Jv4 = np.zeros((3,9))
        Jv4[:, 0] = np.cross(z_0, (o_4 - o_0))
        Jv4[:, 1] = np.cross(z_1, (o_4 - o_1))
        Jv4[:, 2] = np.cross(z_2, (o_4 - o_2))
        Jv4[:, 3] = np.cross(z_3, (o_4 - o_3))
        
        #Jv5
        Jv5 = np.zeros((3,9))
        Jv5[:, 0] = np.cross(z_0, (o_5 - o_0))
        Jv5[:, 1] = np.cross(z_1, (o_5 - o_1))
        Jv5[:, 2] = np.cross(z_2, (o_5 - o_2))
        Jv5[:, 3] = np.cross(z_3, (o_5 - o_3))
        Jv5[:, 4] = np.cross(z_4, (o_5 - o_4))
        
        #Jv6
        Jv6 = np.zeros((3,9))
        Jv6[:, 0] = np.cross(z_0, (o_6 - o_0))
        Jv6[:, 1] = np.cross(z_1, (o_6 - o_1))
        Jv6[:, 2] = np.cross(z_2, (o_6 - o_2))
        Jv6[:, 3] = np.cross(z_3, (o_6 - o_3))
        Jv6[:, 4] = np.cross(z_4, (o_6 - o_4))
        Jv6[:, 5] = np.cross(z_5, (o_6 - o_5))
        
        #Jv_virtual_1
        Jv_virtual_1 = np.zeros((3,9))
        Jv_virtual_1[:, 0] = np.cross(z_0, (o_virtual_1 - o_0))
        Jv_virtual_1[:, 1] = np.cross(z_1, (o_virtual_1 - o_1))
        Jv_virtual_1[:, 2] = np.cross(z_2, (o_virtual_1 - o_2))
        Jv_virtual_1[:, 3] = np.cross(z_3, (o_virtual_1 - o_3))
        Jv_virtual_1[:, 4] = np.cross(z_4, (o_virtual_1 - o_4))
        Jv_virtual_1[:, 5] = np.cross(z_5, (o_virtual_1 - o_5))
        Jv_virtual_1[:, 6] = np.cross(z_6, (o_virtual_1 - o_6))
        Jv_virtual_1[:, 7] = np.cross(z_e, (o_virtual_1 - o_e))
        Jv_virtual_1[:, 8] = np.cross(z_virtual_2, (o_virtual_1 - o_virtual_2))
        
        #Jv_virtual_2
        Jv_virtual_2 = np.zeros((3,9))
        Jv_virtual_2[:, 0] = np.cross(z_0, (o_virtual_2 - o_0))
        Jv_virtual_2[:, 1] = np.cross(z_1, (o_virtual_2 - o_1))
        Jv_virtual_2[:, 2] = np.cross(z_2, (o_virtual_2 - o_2))
        Jv_virtual_2[:, 3] = np.cross(z_3, (o_virtual_2 - o_3))
        Jv_virtual_2[:, 4] = np.cross(z_4, (o_virtual_2 - o_4))
        Jv_virtual_2[:, 5] = np.cross(z_5, (o_virtual_2 - o_5))
        Jv_virtual_2[:, 6] = np.cross(z_6, (o_virtual_2 - o_6))
        Jv_virtual_2[:, 7] = np.cross(z_e, (o_virtual_2 - o_e))
        Jv_virtual_2[:, 8] = np.cross(z_virtual_1, (o_virtual_2 - o_virtual_1))
        
        #9 JOINTS IN F 3x9 ARE 1 2 3 4 5 6 end virtual_1 virtual_2 
        
        #Joint 0 torque 
        F_0 = np.zeros((3, 1))
        torque_0 = Jv0.T @ F_0
        torque_0 = torque_0.reshape(9) #to be consistent with other torques
        
        #Joint 1 torque
        F_1 = joint_forces[:, 0]
        torque_1 = Jv1.T @ F_1
        
        # Joint 2 torque
        F_2 = joint_forces[:, 1]
        torque_2 = Jv2.T @ F_2
        
        # Joint 3 torque
        F_3 = joint_forces[:, 2]
        torque_3 = Jv3.T @ F_3
        
        # Joint 4 torque
        F_4 = joint_forces[:, 3]
        torque_4 = Jv4.T @ F_4
        
        # Joint 5 torque
        F_5 = joint_forces[:, 4]
        torque_5 = Jv5.T @ F_5
        
        # Joint 6 torque
        F_6 = joint_forces[:, 5]
        torque_6 = Jv6.T @ F_6
        
        # Virtual 1 torque
        F_virtual_1 = joint_forces[:, 7]
        torque_virtual_1 = Jv_virtual_1.T @ F_virtual_1
        
        # Virtual 1 torque
        F_virtual_2 = joint_forces[:, 8]
        torque_virtual_2 = Jv_virtual_2.T @ F_virtual_2
        
        # Initialize torques
        joint_torques = np.zeros((1, 9))
        torque_sum = torque_0 + torque_1 + torque_2 + torque_3 + torque_4 + torque_5 + torque_6 + torque_virtual_1 + torque_virtual_2
        joint_torques = torque_sum.reshape(1, 9) #shape back to (1,9) for outputs
       

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        

        distance = 0
        
        distance = np.linalg.norm(target - current)

        

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task. 
        """

       

        dq = np.zeros((1, 7))
        #print(target.shape)
        position_target, _ = PotentialFieldPlanner.fk.forward_expanded(target)
        position_current, _ = PotentialFieldPlanner.fk.forward_expanded(q.reshape(7))
        
        obstacle = map_struct.obstacles
        jointForces = PotentialFieldPlanner.compute_forces(position_target[1:10], obstacle, position_current[1:10]) 
        #from 1:10 since we dont take first position for torque calculation
        
        torques = PotentialFieldPlanner.compute_torques(jointForces, q)
        #take only 7 values since virrtual points have no dq
        torques_7joints = torques[:, :7] - 5 * (q - target) #Introduced additional torque component to make it converge faster
        
        dq = torques_7joints #/np.linalg.norm(torques_7joints) stop division to its norm to check for local minima first
        

        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """

        q_path = np.array([]).reshape(0,7)
        
        #Initialization
        q = start
        iteration = 0
        q_path = np.vstack((start, q_path))
        obstacle = map_struct.obstacles

        while True:
            #Learning rate decreases after each iteration
            alpha = 3/(1+iteration)
            
            
            
            
            
            # Compute gradient 
            # TODO: this is how to change your joint angles 
            
            #Function to check if each of the links within a configuration is valid
            # We check the links between the joints, and the line between the two virtual points 
            #Line between two virtual points is created with joints[8:9] and joints[9:10]
            print("Iteration ", iteration)
            def is_joints_valid(joints, obstacles):
                subset1 = np.concatenate((joints[0:7], joints[8:9]))  # Components 0 to 6 and 8
                subset2 = np.concatenate((joints[1:8], joints[9:10]))
                for obstacle in obstacles: 
                    if True in detectCollision(subset1, subset2, obstacle):
                        return False
                return True

            # Function to check collision through interpolated steps
            def is_collision_free(start_pose, end_pose, obstacles):

                step_size = 0.05
                diff = end_pose - start_pose
                step = np.linalg.norm(diff) / step_size

                for i in range(int(step)):
                    prev_pose = start_pose + i * step_size * diff
                    new_pose = start_pose + (i+1) * step_size * diff

                    prev_joints, _ = PotentialFieldPlanner.fk.forward_expanded(prev_pose)
                    new_joints, _ = PotentialFieldPlanner.fk.forward_expanded(new_pose)

                    if not is_joints_valid(prev_joints, obstacles):
                        return False
                    if not is_joints_valid(new_joints, obstacles):
                        return False

                    for obstacle in obstacles:
                        if True in detectCollision(prev_joints, new_joints, obstacle):
                            return False

                return True
            
            joint_start, transforms_start = PotentialFieldPlanner.fk.forward_expanded(start.reshape(7))
            joint_end, transforms_end = PotentialFieldPlanner.fk.forward_expanded(goal.reshape(7))
            joint_update, transforms_update = PotentialFieldPlanner.fk.forward_expanded(q.reshape(7))
            
            #Check if the start and end position violates limit or obstacles. If it violates, return empty path 
            if np.any((start < self.lower) | (start > self.upper)) or is_joints_valid(joint_start, obstacle) == False or np.any((goal < self.lower) | (goal > self.upper)) or is_joints_valid(joint_end, obstacle) == False:
                q_path = np.array([]).reshape(0,7)
                return q_path
            
            #Loop break conditions: q approach goal enough, or iteration is exceeded
            if (PotentialFieldPlanner.q_distance(goal, q_path[-1]) < self.tol).all() or iteration > self.max_steps: # TODO: check termination conditions
                break # exit the while loop if conditions are met!
            
            # Calculate the gradient of the update
            grad_dq = PotentialFieldPlanner.compute_gradient(q_path[-1], goal, map_struct)
            
            # Update with its grad/norm
            q = q_path[-1] + alpha*(grad_dq/np.linalg.norm(grad_dq))
            
            #Check if valid joints and collision
            valid_joints = is_joints_valid(joint_update, obstacle)
            valid_poses = is_collision_free(q_path[-1].reshape(7), q.reshape(7), obstacle)

            #Initialize the random counter
            random_num = 0
            
            # If joint not valid or there is collision or grad is too small (local minima), we do random walk
            while random_num < 10000000 and (valid_joints == False or valid_poses == False or (np.abs(grad_dq) < 0.0001).all()):
                random_magnitude = np.random.uniform(low=-0.1, high=3.0)
                options = np.array([-random_magnitude, random_magnitude])
                random_sets = np.random.choice(options, size=(7,))
                
                #WHEN UPDATE WITH RANDOM WALK, BASED ON HOW THE RANDOM FUNCTION IS BUILT IN PYTHON
                #IT IS LIKELY THAT A SPECIFIC SEQUENCE OF RANDOM VARIABLES IS GENERATED REPEATEDLY
                # We introduce a second random walk to increase the randomness in case the 1st random walk
                # cannot escape the violated positions
                second_random_sets = np.random.uniform(-random_magnitude, random_magnitude, size=(7,))
                
                # Update q_rand with 1st random set
                q_rand = q_path[-1] + random_sets

                #For each 100 random walks exceeded, we add the additional random walk to guide it into a different path
                if random_num % 100 == 0:
                    q_rand = q_rand + second_random_sets

                q_rand = np.clip(q_rand, self.lower, self.upper) #clip the values to keep it from violating limits

                grad_dq = PotentialFieldPlanner.compute_gradient(q_rand, goal, map_struct) #Calculate new grad
                
                q = q_rand + alpha*(grad_dq/np.linalg.norm(grad_dq)) #New q updated based on the q from random walk
                
                q = np.clip(q, self.lower, self.upper) #Again, clip the values to keep it from violating

                #Calculate new updated positions to make sure the new configurations doesnt violate
                #if it violates, repeat random walk
                joint_update, transforms_update = PotentialFieldPlanner.fk.forward_expanded(q.reshape(7))
                valid_joints = is_joints_valid(joint_update, obstacle)
                valid_poses = is_collision_free(q_path[-1].reshape(7), q.reshape(7), obstacle)
                random_num = random_num + 1
            
            # Clip the final q to add to path
            q = np.clip(q, self.lower, self.upper)
            
            #Add to path and increase iteration count
            q_path = np.vstack((q_path, q))
            iteration = iteration + 1

            
        return q_path
        
        
################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    map_struct = loadmap("../maps/map4.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
    print("error: ", np.linalg.norm(error))

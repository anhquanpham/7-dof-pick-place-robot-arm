import numpy as np
from math import pi

class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout
        self.Tb0 = None
        self.Tb1 = None
        self.Tb2 = None
        self.Tb3 = None
        self.Tb4 = None
        self.Tb5 = None
        self.Tb6 = None
        #pass

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        # Your code starts here
        
        def dh_transform(theta_i, d_i, a_i, alpha_i):
            return np.array([[np.cos(theta_i), -np.sin(theta_i) * np.cos(alpha_i), np.sin(theta_i) * np.sin(alpha_i), a_i * np.cos(theta_i)],
                             [np.sin(theta_i), np.cos(theta_i) * np.cos(alpha_i), -np.cos(theta_i) * np.sin(alpha_i), a_i * np.sin(theta_i)],
                             [0, np.sin(alpha_i), np.cos(alpha_i), d_i],
                             [0, 0, 0, 1]])

        jointPositions = np.zeros((10,3))
        T0e = np.zeros((10,4,4))
        
        q_shown = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
        
        Tb0 = dh_transform(0 , 0.141, 0, 0) #Base to 1st joint
        T01 = dh_transform(0 - q_shown[0] + q[0], 0.192, 0, -np.pi / 2) #1st to second
        T12 = dh_transform(0 - q_shown[1] + q[1], 0, 0, np.pi / 2) #second to third
        T23 = dh_transform(0 - q_shown[2] + q[2], 0.195 + 0.121, 0.0825, np.pi / 2) #Third to fourth just fixed -pi/2
        T34 = dh_transform(np.pi/2 - q_shown[3] + q[3], 0, 0.0825, np.pi / 2) #- 0.825 #Fourth to fifth
        T45 = dh_transform(0 - q_shown[4] + q[4], 0.125 + 0.259, 0,-np.pi/2) #Fifth to sixth
        T56 = dh_transform(-np.pi/2 - q_shown[5] + q[5], 0,0.088,np.pi/2) #Sixth to seventh
        T6e = dh_transform(0 - q_shown[6] + q[6], 0.051 + 0.159, 0, 0) #Seventh to end
        
        self.Tb0 = Tb0
	# Already have 1st joint to base
        # print("Tb0: ", self.Tb0)
        self.Tb1 = self.Tb0 @ T01  # Second joint to base
        # print("Tb1: ", Tb1)
        self.Tb2 = self.Tb1 @ T12  # Third to base
        # print("Tb2: ", Tb2)
        self.Tb3 = self.Tb2 @ T23  # Fourth to base
        # print("Tb3: ", Tb3)
        self.Tb4 = self.Tb3 @ T34  # Fifth to base+
        # print("Tb4: ", Tb4)
        self.Tb5 = self.Tb4 @ T45  #Sixth to base
        # print("Tb5: ", Tb5)
        self.Tb6 = self.Tb5 @ T56 #Seventh to base
        # print("Tb6: ", Tb6)
        Tbe = self.Tb6 @ T6e #End to base
        # print("Tbe: ", Tbe)
        
        # For virtual joints, create new transformation matrices relative to the base
        Tb_virtual_1 = Tbe @ np.array([[1, 0, 0, 0], [0, 1, 0, -0.1], [0, 0, 1, -0.105], [0, 0, 0, 1]])  # Virtual joint 1
        Tb_virtual_2 = Tbe @ np.array([[1, 0, 0, 0], [0, 1, 0, 0.1], [0, 0, 1, -0.105], [0, 0, 0, 1]])  # Virtual joint 2
        
        # Stack the transformations in T0e for each joint
        T0e[0] = self.Tb0  # Tb0 is the transformation from the base to the first joint
        T0e[1] = self.Tb1  # Transformation from base to the second joint
        T0e[2] = self.Tb2  # Transformation from base to the third join
        T0e[3] = self.Tb3  # Transformation from base to the fourth joint
        T0e[4] = self.Tb4  # Transformation from base to the fifth joint
        T0e[5] = self.Tb5  # Transformation from base to the sixth joint
        T0e[6] = self.Tb6  # Transformation from base to the seventh joint
        T0e[7] = Tbe  # Transformation from base to end effector
        
        T0e[8] = Tb_virtual_1  # Virtual joint 1 transformation
        T0e[9] = Tb_virtual_2  # Virtual joint 2 transformation
        
        
        # Extract positions of each joint
        joint1 = self.Tb0 @ np.array([0, 0, 0, 1])
        joint2 = self.Tb1 @ np.array([0, 0, 0, 1])
        joint3 = self.Tb2 @ np.array([0, 0, 0.195, 1])
        joint4 = self.Tb3 @ np.array([0, 0, 0, 1])
        joint5 = self.Tb4 @ np.array([0, 0, 0.125, 1])
        joint6 = self.Tb5 @ np.array([0, 0,-0.015, 1])
        joint7 = self.Tb6 @ np.array([0, 0, 0.051, 1])
        ende = Tbe @ np.array([0,0,0,1])
        
        # Virtual joint positions (relative to end effector frame)
        virtual_joint_1 = Tbe @ np.array([0, -0.1, -0.105, 1])  # Virtual joint 1 (to the left)
        virtual_joint_2 = Tbe @ np.array([0, 0.1, -0.105, 1])  # Virtual joint 2 (to the right)

        jointPositions = np.array([[joint1[0], joint1[1], joint1[2]],
                                   [joint2[0],joint2[1],joint2[2]],
                                   [joint3[0], joint3[1], joint3[2]],
                                   [joint4[0], joint4[1], joint4[2]],
                                   [joint5[0], joint5[1], joint5[2]],
                                   [joint6[0], joint6[1], joint6[2]],
                                   [joint7[0], joint7[1], joint7[2]],
                                   [ende[0], ende[1], ende[2]],
                                   [virtual_joint_1[0],virtual_joint_1[1],virtual_joint_1[2]],
                                   [virtual_joint_2[0],virtual_joint_2[1],virtual_joint_2[2]]])
        
        
                                                 
        

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        

        return()
    
if __name__ == "__main__":

    fk = FK_Jac()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward_expanded(q)
    print("T0e_3", T0e[3][:3, 2])
    
    print("Joint Positions:\n",joint_positions.shape)
    print("End Effector Pose:\n",T0e.shape)

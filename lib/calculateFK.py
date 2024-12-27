import numpy as np
from math import pi, cos, sin

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        self.d = [0.333, 0, 0.316, 0, 0.384, 0, 0.21]
        self.alpha = [-pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2, 0]
        self.a = [0, 0, 0.0825, -0.0825, 0, 0.088, 0]
        self.offset_vectors = [
            np.array([0, 0, 0.141]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0.195]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0.125]),
            np.array([0, 0, -0.015]),
            np.array([0, 0, 0.051]),
            np.array([0,0,0])
        ]

    def dh_transform(self, theta, d, a, alpha):
        return np.array([[cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
                         [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                         [0, sin(alpha), cos(alpha), d],
                         [0, 0, 0, 1]])

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8,3))
        jointPositions[0, :] = self.offset_vectors[0]
        T0e = np.identity(4)

        for i in range(7):
            theta = q[i]
            if i == 6:
                theta -= pi / 4
            T_i = self.dh_transform(theta, self.d[i], self.a[i], self.alpha[i])
            T0e = np.dot(T0e, T_i)

            joint_position_homogeneous = np.dot(T0e, np.append(self.offset_vectors[i+1], 1))
            jointPositions[i+1, :] = joint_position_homogeneous[:3]
        
        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
       

        axis_of_rotation_list = np.zeros((3, 7))
        T0e = np.identity(4)
        
        for i in range(7):
            theta = q[i]
            if i == 6:
                theta -= pi / 4
            T_i = self.dh_transform(theta, self.d[i], self.a[i], self.alpha[i])
            axis_of_rotation_list[:, i] = T0e[:3, 2]
            T0e = np.dot(T0e, T_i)
        
        return axis_of_rotation_list
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        

        Ai = []
        T0e = np.identity(4)
        
        for i in range(7):
            theta = q[i]
            if i == 6:
                theta -= pi / 4
            T_i = self.dh_transform(theta, self.d[i], self.a[i], self.alpha[i])
            T0e = np.dot(T0e, T_i)
            Ai.append(T0e.copy())
    
        return Ai
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    # q = np.array([0, 0, 0, 0, 0, 0, 0])
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    # q = np.array([0,-pi/2,0,-2,0,1,pi/2])

    joint_positions, T0e = fk.forward(q)

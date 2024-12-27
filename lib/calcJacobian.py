import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

   

    fk = FK()
    
    joint_positions, _ = fk.forward(q_in)
    

    axes_of_rotation = fk.get_axis_of_rotation(q_in)

    p_end = joint_positions[-1, :]
    
    for i in range(7):
        p_i = joint_positions[i, :]
        
        z_i = axes_of_rotation[:, i]
        
        Jv_i = np.cross(z_i, p_end - p_i)
        
        Jw_i = z_i
        
        J[:3, i] = Jv_i
        J[3:, i] = Jw_i

    return J


if __name__ == '__main__':
    # q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    q= np.array([0, 0, 0, 0, 0, 0, 0])
    dq = np.array([1, 0, 0, 1, 0, 0, 0])
    J = np.round(calcJacobian(q),3)
    
    # print(J)
    # print(J @ dq)
    

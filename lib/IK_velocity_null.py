import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    
    J = calcJacobian(q_in)
    v_in = np.array(v_in).reshape((3, 1))
    omega_in = np.array(omega_in).reshape((3, 1))
    v_des = np.vstack((v_in, omega_in))

    keep = ~np.isnan(v_des)
    J_constr = J[keep.ravel(), :]

    dq = IK_velocity(q_in, v_in, omega_in)
    J_pseudo = np.linalg.pinv(J_constr) @ J_constr
    N = np.eye(7) - J_pseudo
    null = N @ b
    null = null.reshape((7, 1))
    return dq + null


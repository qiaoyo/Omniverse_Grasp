import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

def rotvector2rot(rotvector):
    Rm = cv2.Rodrigues(rotvector)[0]
    return Rm

def quaternion2euler(quaternion):
    #'xyz'
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler

def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion

def euler2rot(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def rotvector2eular(rotvector):
    Rot = cv2.Rodrigues(rotvector)[0]
    euler=rot2euler(Rot)
    return euler
    
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def quat2rot(quaternion):
    r = R.from_quat(quaternion)
    Rm = r.as_matrix()
    # 0:array([ 1.00000000e+00, -2.74458557e-06,  2.55936079e-06])
    # 1:array([-2.65358979e-06, -3.49007932e-02,  9.99390782e-01])
    # 2:array([-2.65358979e-06, -9.99390782e-01, -3.49007932e-02])

    # 符号相反的四元数, 仍表示同一个旋转
    # Rq1 = [0.71934025092983234, -1.876085535681999e-06, -3.274841213980097e-08, -0.69465790385533299]
    return Rm

def rot2euler(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2]) * 180 / np.pi
        y = math.atan2(-R[2, 0], sy) * 180 / np.pi
        z = math.atan2(R[1, 0], R[0, 0]) * 180 / np.pi
    else:
        x = math.atan2(-R[1, 2], R[1, 1]) * 180 / np.pi
        y = math.atan2(-R[2, 0], sy) * 180 / np.pi
        z = 0

    return np.array([x, y, z])

def rot2quat(Rot):
    r3 = R.from_matrix(Rot)
    quat = r3.as_quat()
    # [0.7193402509298323, -1.8760855356819988e-06, -3.2748412139801076e-08, -0.694657903855333] #与原始相反,但等价
    return quat

if __name__=='__main__':


    rotvector = np.array([[0.223680285784755, 0.240347886848190, 0.176566110650535]])
    print(rotvector2rot(rotvector))

    # 输出
    # [[ 0.95604131 -0.14593404  0.2543389 ]
    #  [ 0.19907538  0.95986385 -0.19756111]
    #  [-0.21529982  0.23950919  0.94672136]]


    quaternion = [0.03551, 0.21960, -0.96928, 0.10494]
    print(quaternion2euler(quaternion))

    # 输出
    # [ -24.90053735    6.599459   -169.1003646 ]


    euler = [-24.90053735, 6.599459, -169.1003646]
    print(euler2quaternion(euler))

    # 输出
    # [ 0.03550998  0.21959986 -0.9692794   0.10493993]
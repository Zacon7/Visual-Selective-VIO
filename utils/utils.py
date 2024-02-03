import os
import glob
import numpy as np
import time
import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import math
from pytorch3d.transforms import matrix_to_quaternion
plt.switch_backend('agg')

_EPS = np.finfo(float).eps * 4.0


def isRotationMatrix(R):
    '''
    check whether a matrix is a qualified rotation metrix
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def euler_from_matrix(matrix):
    '''
    Extract the eular angle from a rotation matrix
    '''
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
    ay = math.atan2(-M[2, 0], cy)
    if ay < -math.pi / 2 + _EPS and ay > -math.pi / 2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2(-M[1, 2], -M[0, 2])
    elif ay < math.pi / 2 + _EPS and ay > math.pi / 2 - _EPS:
        ax = 0
        az = math.atan2(M[1, 2], M[0, 2])
    else:
        ax = math.atan2(M[2, 1], M[2, 2])
        az = math.atan2(M[1, 0], M[0, 0])
    return np.array([ax, ay, az])


def get_relative_pose(Rt1, Rt2):
    '''
    Calculate the relative 4x4 pose matrix between two pose matrices
    '''
    Rt1_inv = np.linalg.inv(Rt1)
    Rt_rel = Rt1_inv @ Rt2
    return Rt_rel


def get_relative_pose_6DoF(Rt1, Rt2):
    '''
    Calculate the relative rotation and translation from two consecutive pose matrices
    '''

    # Calculate the relative transformation Rt_rel
    Rt_rel = get_relative_pose(Rt1, Rt2)

    R_rel = Rt_rel[:3, :3]
    t_rel = Rt_rel[:3, 3]

    # Extract the Eular angle from the relative rotation matrix
    x, y, z = euler_from_matrix(R_rel)
    theta = [x, y, z]

    pose_rel = np.concatenate((theta, t_rel))
    return pose_rel


def rotationError(pose_error):
    '''
    Calculate the rotation difference between two pose matrices
    '''
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))


def translationError(pose_error):
    '''
    Calculate the translational RMSE error between two pose matrices
    '''
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    return np.sqrt(dx**2 + dy**2 + dz**2)


def eulerAnglesToRotationMatrix(theta):
    '''
    Calculate the rotation matrix R from eular angles (roll, yaw, pitch)
    '''
    if torch.is_tensor(theta):
        R_x = torch.tensor([[1, 0, 0],
                            [0, torch.cos(theta[0]), -torch.sin(theta[0])],
                            [0, torch.sin(theta[0]), torch.cos(theta[0])]])
        R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])],
                            [0, 1, 0],
                            [-torch.sin(theta[1]), 0, torch.cos(theta[1])]])
        R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0],
                            [torch.sin(theta[2]), torch.cos(theta[2]), 0],
                            [0, 0, 1]])
        R = torch.matmul(R_z, torch.matmul(R_y, R_x))
    else:
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta[0]), -np.sin(theta[0])],
                        [0, np.sin(theta[0]), np.cos(theta[0])]
                        ])
        R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                        [0, 1, 0],
                        [-np.sin(theta[1]), 0, np.cos(theta[1])]
                        ])
        R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                        [np.sin(theta[2]), np.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def normalize_angle_delta(angle):
    '''
    Normalization angles to constrain that it is between -pi and pi
    '''
    if (angle > np.pi):
        angle = angle - 2 * np.pi
    elif (angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle


def pose_6DoF_to_matrix(pose):
    '''
    Calculate the 4x4 transformation matrix SE(3) from Eular angles and translation vector R6
    '''
    if torch.is_tensor(pose):
        R = eulerAnglesToRotationMatrix(pose[:3])
        t = pose[3:].view(3, 1)
        R = torch.cat((R, t), 1)
        R = torch.cat((R, torch.tensor([[0, 0, 0, 1]])), 0)
        return R
    else:
        R = eulerAnglesToRotationMatrix(pose[:3])
        t = pose[3:].reshape(3, 1)
        R = np.concatenate((R, t), 1)
        R = np.concatenate((R, np.array([[0, 0, 0, 1]])), 0)
    return R


def pose_accu(Rt_pre, R6_rel):
    '''
    Calculate the accumulated pose from the latest pose and the relative rotation and translation
    '''
    Rt_rel = pose_6DoF_to_matrix(R6_rel)
    return Rt_pre @ Rt_rel


def path_accu(rel_pose):
    '''
    Generate the global pose matrices from a series of relative poses
    '''
    if torch.is_tensor(rel_pose):
        global_poses = [torch.eye(4)]
        for index in range(rel_pose.shape[0]):
            curr_pose = pose_accu(global_poses[-1], rel_pose[index, :])
            global_poses.append(curr_pose)
        return torch.stack(global_poses, dim=0)
    else:
        global_poses = [np.eye(4)]
        for index in range(rel_pose.shape[0]):
            curr_pose = pose_accu(global_poses[-1], rel_pose[index, :])
            global_poses.append(curr_pose)
        return global_poses


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def rmse_err_cal(rel_pose_est, rel_pose_gt):
    '''
    Calculate the rmse of relative translation and rotation
    '''
    t_rmse = np.sqrt(np.mean(np.sum((rel_pose_est[:, 3:] - rel_pose_gt[:, 3:])**2, -1)))
    r_rmse = np.sqrt(np.mean(np.sum((rel_pose_est[:, :3] - rel_pose_gt[:, :3])**2, -1)))
    return t_rmse, r_rmse


def trajectoryDistances(global_poses):
    '''
    Calculate the distance and speed for each frame
    '''
    dists = [0]
    speeds = [0]
    for i in range(len(global_poses) - 1):
        cur_frame_idx = i
        next_frame_idx = cur_frame_idx + 1
        P1 = global_poses[cur_frame_idx]
        P2 = global_poses[next_frame_idx]
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dz = P1[2, 3] - P2[2, 3]
        dists.append(dists[i] + np.sqrt(dx**2 + dy**2 + dz**2))
        speeds.append(np.sqrt(dx**2 + dy**2 + dz**2) * 10)
    return dists, speeds


def lastFrameFromSegmentLength(dist, first_frame, len_):
    for i in range(first_frame, len(dist), 1):
        if dist[i] > (dist[first_frame] + len_):
            return i
    return -1


def computeOverallErr(seq_err):
    '''
    Calculate average errors per meter for all subsequences of different lengths
    '''
    t_err = 0
    r_err = 0
    seq_len = len(seq_err)

    for item in seq_err:
        r_err += item[1]
        t_err += item[2]
    ave_t_err = t_err / seq_len
    ave_r_err = r_err / seq_len
    return ave_t_err, ave_r_err


def read_pose(line):
    '''
    Reading 4x4 pose matrix from .txt files
    input: a line of 12 parameters
    output: 4x4 numpy matrix
    '''
    values = np.reshape(np.array([float(value) for value in line.split(' ')]), (3, 4))
    Rt = np.concatenate((values, np.array([[0, 0, 0, 1]])), 0)
    return Rt


def read_pose_from_text(path):
    '''
    Reading abs pose(SE3) and rel pose (R6) from text file
    '''
    with open(path) as f:
        lines = [line.split('\n')[0] for line in f.readlines()]
        poses_rel, poses_abs = [], []
        prev_pose = read_pose(lines[0])
        poses_abs.append(prev_pose)
        for i in range(1, len(lines)):
            curr_pose = read_pose(lines[i])
            poses_rel.append(get_relative_pose_6DoF(prev_pose, curr_pose))
            prev_pose = curr_pose.copy()
            poses_abs.append(curr_pose)
        poses_abs = np.array(poses_abs)
        poses_rel = np.array(poses_rel)
    return poses_abs, poses_rel


def poses_SE3_to_quaternion(poses):
    '''
    Convert a SE(3) Pose to Quaternion Pose
    input:  poses: (n, 4, 4) in SE(3)
    output: poses_qua: (n, 4) in quaternion
    '''
    rot_mat = torch.tensor(poses[:, :3, :3])
    poses_qua = matrix_to_quaternion(rot_mat).numpy()
    return poses_qua


def saveSequence(poses, file_name):
    with open(file_name, 'w') as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            f.write(' '.join([str(r) for r in pose]))
            f.write('\n')


# if __name__ == '__main__':
#     R = np.array([[-0.545, 0.797, 0.260, 0],
#                   [0.733, 0.603, -0.313, 0],
#                   [-0.407, 0.021, -0.913, 0],
#                   [0, 0, 0, 1]])
#     R = np.array([[0.395, 0.362, 0.843, 0],
#                   [-0.626, 0.796, -0.056, 0],
#                   [-0.677, -0.498, 0.529, 0],
#                   [0, 0, 0, 1]])

#     q1 = matrix_to_quaternion(torch.tensor(R[:3, :3])).numpy()
#     q2 = from_rotation_matrix(R[:3, :3], False)

#     print(q1)
#     print(q2)

# qua_pred[i,]=w0,x0,y0,z0
# qua_true[i,]=w1,x1,y1,z1
# w0w1 − x0x1 − y0y1 − z0z1
# w0x1 + x0w1 + y0z1 − z0y1
# w0y1 − x0z1 + y0w1 + z0x1
# w0z1 + x0y1 − y0x1 + z0w1

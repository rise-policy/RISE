"""
Transformation utilities.
"""

from typing import Union

import torch
import functools
import numpy as np
import pytorch3d.transforms.rotation_conversions as ptc

from utils import rotation_utils as rtu

VALID_ROTATION_REPRESENTATIONS = [
    'axis_angle',
    'euler_angles',
    'quaternion',
    'matrix',
    'rotation_6d',
    'rotation_9d',
    'rotation_10d'
]
ROTATION_REPRESENTATION_DIMS = {
    'axis_angle': 3,
    'euler_angles': 3,
    'quaternion': 4,
    'matrix': 9,
    'rotation_6d': 6,
    'rotation_9d': 9,
    'rotation_10d': 10
}


def rotation_transform(
    rot,
    from_rep, 
    to_rep, 
    from_convention = None, 
    to_convention = None
):
    """
    Transform a rotation representation into another equivalent rotation representation.
    """
    assert from_rep in VALID_ROTATION_REPRESENTATIONS, "Invalid rotation representation: {}".format(from_rep)
    assert to_rep in VALID_ROTATION_REPRESENTATIONS, "Invalid rotation representation: {}".format(to_rep)
    if from_rep == 'euler_angles':
        assert from_convention is not None
    else:
        from_convention = None
    if to_rep == 'euler_angles':
        assert to_convention is not None
    else:
        to_convention = None

    if from_rep == to_rep and from_convention == to_convention:
        return rot

    if from_rep != "matrix":
        if from_rep in ['rotation_9d', 'rotation_10d']:
            to_mat = getattr(rtu, "{}_to_matrix".format(from_rep))
        else:
            to_mat = getattr(ptc, "{}_to_matrix".format(from_rep))
            if from_convention is not None:
                to_mat = functools.partial(to_mat, convention = from_convention)
        mat = to_mat(torch.from_numpy(rot)).numpy()
    else:
        mat = rot
        
    if to_rep != "matrix":
        if to_rep in ['rotation_9d', 'rotation_10d']:
            to_ret = getattr(rtu, "matrix_to_{}".format(to_rep))
        else:
            to_ret = getattr(ptc, "matrix_to_{}".format(to_rep))
            if to_convention is not None:
                to_ret = functools.partial(to_ret, convention = to_convention)
        ret = to_ret(torch.from_numpy(mat)).numpy()
    else:
        ret = mat
    
    return ret


def xyz_rot_transform(
    xyz_rot,
    from_rep, 
    to_rep, 
    from_convention = None, 
    to_convention = None
):
    """
    Transform an xyz_rot representation into another equivalent xyz_rot representation.
    """
    assert from_rep in VALID_ROTATION_REPRESENTATIONS, "Invalid rotation representation: {}".format(from_rep)
    assert to_rep in VALID_ROTATION_REPRESENTATIONS, "Invalid rotation representation: {}".format(to_rep)

    if from_rep == to_rep and from_convention == to_convention:
        return xyz_rot

    xyz_rot = np.array(xyz_rot)
    if from_rep != "matrix":
        assert xyz_rot.shape[-1] == 3 + ROTATION_REPRESENTATION_DIMS[from_rep]
        xyz = xyz_rot[..., :3]
        rot = xyz_rot[..., 3:]
    else:
        assert xyz_rot.shape[-1] == 4 and xyz_rot.shape[-2] == 4
        xyz = xyz_rot[..., :3, 3]
        rot = xyz_rot[..., :3, :3]
    rot = rotation_transform(
        rot = rot,
        from_rep = from_rep,
        to_rep = to_rep,
        from_convention = from_convention,
        to_convention = to_convention
    )
    if to_rep != "matrix":
        return np.concatenate((xyz, rot), axis = -1)
    else:
        res = np.zeros(xyz.shape[:-1] + (4, 4), dtype = np.float32)
        res[..., :3, :3] = rot
        res[..., :3, 3] = xyz
        res[..., 3, 3] = 1
        return res


def xyz_rot_to_mat(xyz_rot, rotation_rep, rotation_rep_convention = None):
    """
    Transform an xyz_rot representation under any rotation form to an unified 4x4 pose representation.
    """
    return xyz_rot_transform(
        xyz_rot,
        from_rep = rotation_rep,
        to_rep = "matrix",
        from_convention = rotation_rep_convention
    )


def mat_to_xyz_rot(mat, rotation_rep, rotation_rep_convention = None):
    """
    Transform an unified 4x4 pose representation to an xyz_rot representation under any rotation form.
    """
    return xyz_rot_transform(
        mat,
        from_rep = "matrix",
        to_rep = rotation_rep,
        to_convention = rotation_rep_convention
    )


def apply_mat_to_pose(pose, mat, rotation_rep, rotation_rep_convention = None):
    """
    Apply transformation matrix mat to pose under any rotation form.
    """
    assert rotation_rep in VALID_ROTATION_REPRESENTATIONS, "Invalid rotation representation: {}".format(rotation_rep)
    mat = np.array(mat)
    pose = np.array(pose)
    assert mat.shape == (4, 4)
    if rotation_rep == "matrix":
        assert pose.shape[-2] == 4 and pose.shape[-1] == 4
        res_pose = mat @ pose
        return res_pose
    assert pose.shape[-1] == 3 + ROTATION_REPRESENTATION_DIMS[rotation_rep]
    pose_mat = xyz_rot_to_mat(
        xyz_rot = pose,
        rotation_rep = rotation_rep,
        rotation_rep_convention = rotation_rep_convention
    )
    res_pose_mat = mat @ pose_mat
    res_pose = mat_to_xyz_rot(
        mat = res_pose_mat,
        rotation_rep = rotation_rep,
        rotation_rep_convention = rotation_rep_convention
    )
    return res_pose


def apply_mat_to_pcd(pcd, mat):
    """
    Apply transformation matrix mat to point cloud.
    """
    mat = np.array(mat)
    assert mat.shape == (4, 4)
    pcd[..., :3] = (mat[:3, :3] @ pcd[..., :3].T).T + mat[:3, 3]
    return pcd


def rot_mat_x_axis(angle):
    """
    3x3 transformation matrix for rotation along x axis.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype = np.float32)

def rot_mat_y_axis(angle):
    """
    3x3 transformation matrix for rotation along y axis.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype = np.float32)

def rot_mat_z_axis(angle):
    """
    3x3 transformation matrix for rotation along z axis.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype = np.float32)

def rot_mat(angles):
    """
    3x3 transformation matrix for rotation along x, y, z axes.
    """
    x_mat = rot_mat_x_axis(angles[0])
    y_mat = rot_mat_y_axis(angles[1])
    z_mat = rot_mat_z_axis(angles[2])
    return z_mat @ y_mat @ x_mat

def trans_mat(offsets):
    """
    4x4 transformation matrix for translation along x, y, z axes.
    """
    res = np.identity(4, dtype = np.float32)
    res[:3, 3] = np.array(offsets)
    return res

def rot_trans_mat(offsets, angles):
    """
    4x4 transformation matrix for rotation along x, y, z axes, then translation along x, y, z axes.
    """
    res = np.identity(4, dtype = np.float32)
    res[:3, :3] = rot_mat(angles)
    res[:3, 3] = np.array(offsets)
    return res

def trans_rot_mat(offsets, angles):
    """
    4x4 transformation matrix for translation along x, y, z axes, then rotation along x, y, z axes.
    """
    res = np.identity(4, dtype = np.float32)
    res[:3, :3] = rot_mat(angles)
    offsets = (res[:3, :3] @ np.array(offsets).unsqueeze(-1)).squeeze()
    res[:3, 3] = offsets
    return res

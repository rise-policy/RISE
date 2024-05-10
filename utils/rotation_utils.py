"""
(Extended) Conversion functions between rotation representations (9D, 10D) and rotation matrix.

References:
- rotation 9d: Levinson et al, An Analysis of SVD for Deep Rotation Estimation, NeurIPS 2020.
               https://github.com/amakadia/svd_for_pose
- rotation 10d: Peretroukhin et al, A Smooth Representation of SO(3) for Deep Rotation Learning with Uncertainty, RSS 2020.
                https://github.com/utiasSTARS/bingham-rotation-learning
"""

import torch
import pytorch3d.transforms.rotation_conversions as ptc


def rotation_9d_to_matrix(rotation_9d):
    """
    Map 9D input vectors onto SO(3) rotation matrix.
    """
    batch_dim = rotation_9d.size()[:-1]
    m = rotation_9d.view(batch_dim + (3, 3))
    u, s, vt = torch.linalg.svd(m, full_matrices = False)
    det = torch.det(u @ vt)
    det = det.view(batch_dim + (1, 1))
    vt = torch.cat((vt[..., :2, :], vt[..., -1:, :] * det), dim = -2)
    r = u @ vt
    return r


def matrix_to_rotation_9d(matrix):
    """
    Map rotation matrix to 9D rotation representation. The mapping is not unique.

    Note that the rotation matrix itself is a valid 9D rotation representation.
    """
    return matrix


def rotation_10d_to_matrix(rotation_10d):
    """
    Map 10D input vectors to SO(3) rotation matrix.
    """
    batch_dim = rotation_10d.size()[:-1]
    idx = torch.triu_indices(4, 4)
    A = rotation_10d.new_zeros(batch_dim + (4, 4))
    A[..., idx[0], idx[1]] = rotation_10d
    A[..., idx[1], idx[0]] = rotation_10d
    _, evs = torch.linalg.eigh(A, UPLO = 'U')
    quat = evs[..., 0]
    matrix = ptc.quaternion_to_matrix(quat)
    return matrix


def matrix_to_rotation_10d(matrix):
    """
    Map rotation matrix to 10D rotation representation. The mapping is not unique.
    
    See: https://github.com/utiasSTARS/bingham-rotation-learning/issues/8
    """
    batch_dim = matrix.size()[:-2]
    quat = ptc.matrix_to_quaternion(matrix)
    A = torch.eye(4).repeat(batch_dim + (1, 1)).type(quat.dtype).to(quat.device) - quat.unsqueeze(-1) @ quat.unsqueeze(-2)
    idx = torch.triu_indices(4, 4)
    rotation_10d = A[..., idx[0], idx[1]]
    return rotation_10d

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Optional
import torch
import torch.nn.functional as F

"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :6].clone().reshape(*matrix.size()[:-2], 6)


def transformSimplify(matrix: torch.Tensor) -> torch.Tensor:
    """
    convert ???*16 or ???*4*4 matrix to ???*9 matrix
    """
    if matrix.shape[-1] == 16:
        matrix = matrix.view(*matrix.shape[:-1], 4, 4)
    assert(matrix.shape[-1] == 4)
    rotation = matrix_to_rotation_6d(matrix[...,:2,:3]) # return ???*6 matrix
    return torch.cat([rotation, matrix[...,:3,3]],dim=-1).to(matrix.device)



def transformRestore(matrix: torch.Tensor) -> torch.Tensor:
    """
    convert ???*9 or ???*3*3 matrix ???*4*4 matrix
    """
    if matrix.shape[-1] == 3:
        matrix = matrix.view(*matrix.shape[:-2], -1)
    assert(matrix.shape[-1] == 9)
    result = torch.tile(torch.eye(4),dims=(*matrix.shape[:-1],1,1))
    result[...,:3,:3] = rotation_6d_to_matrix(matrix[...,:6])
    result[...,:3,3] = matrix[...,6:]


    return result.to(matrix.device)

def translationToTransform(tranlation: torch.Tensor,
                           simplify_matrix=True) -> torch.Tensor:
    """
        return: 
            if set simplify_matrix as Fasle, it will convert ???*3 translation to ???*9 matrix
            if set simplify_matrix as True, it will convert ???*3 translation to ???*4*4 matrix

    """
    if simplify_matrix:
        rotation = torch.zeros([*tranlation.shape[:-1], 6], dtype=torch.float32).to(tranlation.device)
        rotation[...,[0,4]] = 1.
        return torch.cat([rotation,tranlation], dim=-1)
    else:
        result = torch.tile(torch.eye(4,dtype=torch.float32),dims=(*tranlation.shape[:-1],1,1))
        result[...,:3,3] = tranlation

        return result.to(tranlation.device)


def absoluteToRelativeTransform(matrix: torch.Tensor):
    """
    convert k*16 or k*4*4 absolute transformation to (k-1)*16 relative transformation
    """
    if matrix.shape[-1] == 16:
        matrix = matrix.view(1,4,4)
    
    matrix_inverse = matrix.inverse()
    prev_ind = 0
    relative_transform = torch.empty([matrix.shape[0]-1, 4,4], dtype = matrix.dtype)
    for cur_ind in range(0,matrix.shape[0]):
        prev_ind = cur_ind-1
        relative_transform[prev_ind] = torch.einsum('ijk,ikl->ijl',matrix[cur_ind], matrix_inverse[prev_ind])

    return relative_transform.view(-1,16)

def getDerivative(X, n, dt):
    for i in range(n):
        X = (X[1:] - X[:-1]) / dt
    return X


    
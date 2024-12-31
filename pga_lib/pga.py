
import os
import sys
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ''))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../../'))

from quaternions import quaternion_from_angle_axis

global global_var 

global_var = {
    # Geometric algebra
    'ga_dimension': 16,
    'grade_components': [1, 4, 6, 4, 1],
}

def embed_point(coordinates: torch.Tensor) -> torch.Tensor:
    """Embeds 3D points in multivectors.

    We follow the convention used in the reference below and map points to tri-vectors.

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
    https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    coordinates : torch.Tensor with shape (..., 3)
        3D coordinates

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = coordinates.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=coordinates.dtype, device=coordinates.device)

    # Embedding into trivectors
    # Homogeneous coordinates: unphysical component / embedding dim, x_123
    multivector[..., 14] = 1.0
    multivector[..., 13] = -coordinates[..., 0]  # x-coordinate embedded in x_023
    multivector[..., 12] = coordinates[..., 1]  # y-coordinate embedded in x_013
    multivector[..., 11] = -coordinates[..., 2]  # z-coordinate embedded in x_012

    return multivector


def get_translation_mv(translation: torch.Tensor) -> torch.Tensor:
    """
    Generates the multivector representations for translation and its inverse.

    Parameters:
        translation (List[float]): List containing translation values along x, y, and z axes.

    Returns:
        (Tuple[torch.Tensor]): Multivector representations for translation and its inverse.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = translation.shape[:-1]
    translation_mv = torch.zeros(
        *batch_shape, 16, dtype=translation.dtype, device=translation.device
    )

    inv_translation = list(map(lambda x: -x, translation))

    # [batch, item, channels, mv_dim]
    translation_mv = torch.zeros(1,1,1,global_var['ga_dimension'])
    translation_mv[..., 0] = 1
    translation_mv[..., 5] = 0.5 * translation[0]
    translation_mv[..., 6] = 0.5 * translation[1]
    translation_mv[..., 7] = 0.5 * translation[2]

    inv_translation_mv = torch.zeros(1,1,1,global_var['ga_dimension'])
    inv_translation_mv[..., 0] = 1
    inv_translation_mv[..., 5] = 0.5 * inv_translation[0]
    inv_translation_mv[..., 6] = 0.5 * inv_translation[1]
    inv_translation_mv[..., 7] = 0.5 * inv_translation[2]

    return translation_mv, inv_translation_mv


def get_rotation_mv(rotation: torch.Tensor, angle, axis):
    """
    Generates the multivector representations for rotation and its inverse.

    Parameters:
        angle (float): Rotation angle in degrees.
        axis (np.ndarray): Axis of rotation represented as a 3D vector.

    Returns:
        (Tuple[torch.Tensor]): Multivector representations for rotation and its inverse.
    """

    quaternion = quaternion_from_angle_axis(
        alpha = angle,
        axis = axis
    )

    rotation_quaternion = quaternion.quaternion_to_numpy()
    inv_rotation_quaternion = quaternion.conjugate().quaternion_to_numpy()

    # [batch,item,channels,mv_dim]
    # rotation_mv = torch.zeros(1,1,1,global_var['ga_dimension'])

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = rotation.shape[:-1]
    rotation_mv = torch.zeros(
        *batch_shape, 16, dtype=rotation.dtype, device=rotation.device
    )

    rotation_mv[..., 0] = rotation_quaternion[0]
    rotation_mv[..., 8] = rotation_quaternion[1]
    rotation_mv[..., 9] = rotation_quaternion[2]
    rotation_mv[..., 10] = rotation_quaternion[3]

    inv_rotation_mv = torch.zeros(1,1,1,global_var['ga_dimension'])

    inv_rotation_mv[..., 0] = inv_rotation_quaternion[0]
    inv_rotation_mv[..., 8] = inv_rotation_quaternion[1]
    inv_rotation_mv[..., 9] = inv_rotation_quaternion[2]
    inv_rotation_mv[..., 10] = inv_rotation_quaternion[3]

    return rotation_mv, inv_rotation_mv

def blade_operator():
    """
    Generates a blade operator matrix for the geometric algebra

    Args:
        None

    Returns:
        (torch.Tensor): Blade operator matrix.
    """

    mv_dimension = global_var['ga_dimension']
    blade_shape = (mv_dimension,mv_dimension)

    coordinates = []
    start = 0
    for length in global_var['grade_components']:
        coordinates.append(list(range(start, start + length)))
        start += length

    coord_permutations = [
        [[0,1]],
        [[2,5],[3,6],[4,7]],
        [[8,11],[9,12],[10,13]],
        [[14,15]]
     ]
    blade_mask = []

    w_dimension = len(global_var['grade_components'])
    for k_grade in range(w_dimension):
        w_blade = torch.zeros(blade_shape)
        for coordinate in coordinates[k_grade]:
            w_blade[coordinate, coordinate] = 1.0
        blade_mask.append(w_blade.unsqueeze(0))

    v_dimension = len(global_var['grade_components']) - 1
    for k_grade in range(v_dimension):
        v_blade = torch.zeros(blade_shape)
        for coord_to,coord_from in coord_permutations[k_grade]:
            v_blade[coord_from, coord_to] = 1.0
        blade_mask.append(v_blade.unsqueeze(0))

    blade_operator = torch.cat(blade_mask,dim = 0)

    return blade_operator


def get_coordinates_range():
    """
    Get the ranges of coordinates for each grade based on the configuration in `global_var`.

    Args:
        None

    Returns:
        (List[List[int]]): List of coordinate ranges for each grade.
    """
    grade_components = global_var['grade_components']
    coordinates_range = []

    for grade in range(len(grade_components)):
        start_idx = sum(grade_components[:grade])
        end_idx = sum(grade_components[:grade + 1]) - 1
        coordinate_range = [start_idx,end_idx]
        coordinates_range.append(coordinate_range)

    return coordinates_range


def reverse_operator():
    """
    Generate a reverse operator for the given multivector configuration.

    Args:
        None

    Returns:
        (torch.Tensor): Reverse operator for the multivector space.
    """

    reverse_operator = torch.ones(16)
    *_, bivector_range, trivector_range, _ = get_coordinates_range()

    reverse_range = list(
        range(
            bivector_range[0],
            trivector_range[-1] + 1
         )
     )

    reverse_operator[reverse_range] = -1

    return reverse_operator.to('cuda')


def dual_operators():
    """
    Constructs dual operators, including the indices for sign flipping and the sign values.

    Args:
        None

    Returns:
        (Tuple[List[int], torch.Tensor]): Tuple containing a list of indices to mask to perfom dual and
                                          the vector to perfom it. This masked approach make faster the
                                          dual computation when applied to a multivector
    """

    coords_range = get_coordinates_range()

    dual_sign_idxs = [2, 4, 6, 9, 12, 14]
    dual_signs = [
        1 if i not in dual_sign_idxs else -1
        for i in range(global_var['ga_dimension'])
    ][::-1]
    dual_signs = torch.tensor(dual_signs)

    dual_flip = list(
        range(
            coords_range[-1][0], coords_range[0][0]-1,-1
        )
     )

    return dual_flip, dual_signs.to('cuda')

def compute_dualization(device=torch.device("cuda"), dtype=torch.float32) :
    """Constructs a tensor for the dual operation.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    permutation : list of int
        Permutation index list to compute the dual
    factors : torch.Tensor
        Signs to multiply the dual outputs with.
    """
    permutation = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    factors = torch.tensor(
        [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1], device=device, dtype=dtype
    )
    return permutation, factors


def get_guidance_matrix():
    """
    Fetches and loads the guidance matrix from a specified URL,
    saving it locally if not already present.

    Returns:
        (torch.Tensor): A tensor representing the guidance matrix.
    """

    guidance_matrix = product_basis = torch.load(os.path.join(BASE_DIR, "data", "geometric_product.pt"), weights_only=True)
    guidance_matrix = guidance_matrix.to(torch.float32)
    guidance_matrix = guidance_matrix.to_dense()

    return guidance_matrix

def geometric_product(x, y):
    """
    Computes the geometric product of two multivectors using the guidance matrix.

    Args:
        x (torch.Tensor): the first multivector
        y (torch.Tensor): the second multivector

    Returns:
        (torch.Tensor): multivector result of the geometric product
    """

    guidance_matrix = get_guidance_matrix().to('cuda')

    geom_prod = torch.einsum(
        "i j k, ... j, ... k -> ... i",
        guidance_matrix,
        x,
        y
     )

    return geom_prod


def faster_inner_product(x, y):
    """
    Computes the inner product of two multivectors using a faster method.

    Args:
        x (torch.Tensor): the first multivector
        y (torch.Tensor): the second multivector

    Returns:
        (torch.Tensor): scalar result of the inner product
    """
    guidance_matrix = get_guidance_matrix().to('cuda')
    reverse_op = reverse_operator()

    inner_product_mask = (
        torch.diag(guidance_matrix[0]) * reverse_op
    ).bool()

    x = x[..., inner_product_mask]
    y = y[..., inner_product_mask]

    # questo è un "geom prod2 modificato"
    inner_product = torch.einsum(
        "... i, ... i -> ...", x, y
    ).unsqueeze(-1)

    return inner_product


def get_outer_matrix():
    """
    Fetches and loads the outer product guidance matrix
    from a specified URL,  saving it locally if not already present.

    Returns:
        (torch.Tensor): A tensor representing the outer guidance matrix.
    """

    outer_matrix = torch.load(os.path.join(BASE_DIR, "data", "outer_product.pt"), weights_only=True)
    outer_matrix = outer_matrix.to(torch.float32)
    outer_matrix = outer_matrix.to_dense()

    return outer_matrix


def outer_product(x, y):
    """
    Computes the outer product of two multivectors using the outer guidance matrix.

    Args:
        x (torch.Tensor): the first multivector
        y (torch.Tensor): the second multivector

    Returns:
        (torch.Tensor): multivector result of the outer product
    """

    outer_matrix = get_outer_matrix().to('cuda')

    outputs = torch.einsum("i j k, ... j, ... k -> ... i", outer_matrix, x, y)
    return outputs


def join(x, y, ref):
    """
    Computes the join of two multivectors relative to a reference multivector.

    Args:
        x (torch.Tensor): the first multivector
        y (torch.Tensor): the second multivector
        ref (torch.Tensor): the reference multivector, mean over the batch
                            and channels of the entering GATr multivector

    Returns:
        (torch.Tensor): multivector result of the join operation
    """

    dual_flip, dual_signs = dual_operators()
    dual_x = dual_signs * x[..., dual_flip]
    dual_y = dual_signs * y[..., dual_flip]

    # print(dual_x)
    # print(dual_y)

    outer_prod = outer_product(dual_x, dual_y)
    classic_join = dual_signs * outer_prod[...,dual_flip]
    equi_join = ref[..., [15]] * classic_join

    return equi_join


def grade_involution(mv):
    """
    Applies the grade involution to a multivector, flipping the signs of odd-graded components.

    Args:
        mv (torch.Tensor): the input multivector

    Returns:
        (torch.Tensor): multivector result of the grade involution
        (list): indices of flipped signs during the involution
    """

    odd_grades = get_coordinates_range()[1::2]
    flip_signs = []

    for grade_range in odd_grades:
        flip_signs += list(range(grade_range[0],grade_range[1] + 1))

    involution_signs = [-1 if i in flip_signs else 1 for i in range(global_var['ga_dimension'])][::-1]
    involution_signs = torch.tensor(involution_signs).to('cuda')

    involuted_mv = involution_signs * mv

    return involuted_mv, flip_signs


def sandwich_product(mv,u,inv_u):
    """
    Computes the sandwich product of a multivector and a matrix.

    Args:
        mv (torch.Tensor): the input multivector
        u (torch.Tensor): the matrix for the sandwich product

    Returns:
        (torch.Tensor): a torch tensor representing the result
                        of the sandwich product
    """
    first_geom_product = geometric_product(u,mv)
    output = geometric_product(first_geom_product,inv_u)

    return output
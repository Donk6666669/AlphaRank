"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.

Hyperbolic geometry has a direct connection to the study of special relativity
theory -- implementations in this module borrow some of its terminology. The axis
of symmetry of the Hyperboloid is called the _time dimension_, while all other
axes are collectively called _space dimensions_.

All functions implemented here only input/output the space components, while
while calculating the time component according to the Hyperboloid constraint:

    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from loguru import logger


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
        between input vectors.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl
def lorentz_distance(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """
    Compute Lorentzian distance between two sets of points on the hyperboloid.

    Formula:
        d_L(x, y) = 2 / curv - 2 * <x, y>_L

    Args:
        x: Tensor of shape (B1, D) representing first batch of points.
        y: Tensor of shape (B2, D) representing second batch of points.
        curv: Positive scalar denoting the (negative) curvature magnitude.

    Returns:
        Tensor of shape (B1, B2) giving Lorentzian distances.
    """
    xyl = pairwise_inner(x, y, curv=curv)
    dist = 2.0 / curv - 2.0 * xyl
    return dist

def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
        connecting the input points.
    """

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


# def pairwise_sinh(
#     x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
# ) -> torch.Tensor:
#     """
#     Compute sinh(d) for corresponding pairs of Lorentz points (1-to-1).
#     """

#     # Lorentzian inner product term (逐元素)
#     c_xyl = -torch.sum(x * y, dim=-1)  # shape: (B,)

#     # 计算 cosh(d)
#     cosh_d = torch.clamp(c_xyl, min=1 + eps)

#     # 计算 sinh(d)
#     sinh_d = torch.sqrt(torch.clamp(cosh_d**2 - 1.0, min=eps))

#     return sinh_d
def pairwise_sinh(x, y, eps=1e-6):
    c_xyl = -torch.sum(x * y, dim=-1)
    c_xyl = torch.clamp(c_xyl, min=1.0 + eps)
    sinh_d = torch.sqrt(torch.clamp(c_xyl**2 - 1.0, min=eps))
    return sinh_d



# def pairwise_dist(x: Tensor, y: Tensor, curv: float = 1.0, eps: float = 1e-8) -> Tensor:
#     # x, y: (B, D+1) full Lorentz vectors
#     # Lorentz inner product
#     def lorentz_inner(a, b):
#         return -a[..., 0:1] * b[..., 0:1] + torch.sum(a[..., 1:] * b[..., 1:], dim=-1, keepdim=True)

#     c_xyl = -curv * lorentz_inner(x.unsqueeze(1), y.unsqueeze(0)).squeeze(-1)  # (B1, B2)

#     return torch.acosh(torch.clamp(c_xyl, min=1.0 + eps)) / curv**0.5


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid, on to the
    hyperboloid. This mapping is done using the exponential map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving batch of Euclidean vectors to project
            onto the hyperboloid. These vectors are interpreted as velocity
            vectors in the tangent space at the hyperboloid vertex.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output

# def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
#     rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

#     # clamp 避免溢出
#     norm_clamped = torch.clamp(rc_xnorm, min=eps, max=15.0)  # 限制到 [eps, 15]

#     x0 = torch.cosh(norm_clamped)
#     xi = torch.sinh(norm_clamped) * x / torch.clamp(rc_xnorm, min=eps)

#     return torch.cat([x0, xi], dim=-1)  # (B, D+1)


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """

    # Calculate distance of vectors to the hyperboloid vertex.
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid. The given vector would meet the apex of this cone, and the
    cone itself extends outwards to infinity.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        min_radius: Radius of a small neighborhood around vertex of the hyperboloid
            where cone aperture is left undefined. Input vectors lying inside this
            neighborhood (having smaller norm) will be projected on the boundary.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the half-aperture of entailment cones
        formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
    """

    # Ensure numerical stability in arc-sin by clamping input.
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle
# def oxy_angle(x: Tensor, y: Tensor, curv: float = 1.0, eps: float = 1e-8):
#     # x, y: (B, D+1) full Lorentz vectors
#     def lorentz_inner(a, b):
#         return -a[..., 0] * b[..., 0] + torch.sum(a[..., 1:] * b[..., 1:], dim=-1)

#     dOx = torch.acosh(torch.clamp(-curv * lorentz_inner(x, torch.tensor([1.0] + [0.0]*(x.size(-1)-1), device=x.device, dtype=x.dtype)), min=1+eps)) / curv**0.5
#     dOy = torch.acosh(torch.clamp(-curv * lorentz_inner(y, torch.tensor([1.0] + [0.0]*(y.size(-1)-1), device=y.device, dtype=y.dtype)), min=1+eps)) / curv**0.5
#     dxy = torch.acosh(torch.clamp(-curv * lorentz_inner(x, y), min=1+eps)) / curv**0.5

#     cos_angle = (torch.cosh(dOy*curv**0.5) - torch.cosh(dOx*curv**0.5) * torch.cosh(dxy*curv**0.5)) / (
#         torch.sinh(dOx*curv**0.5) * torch.sinh(dxy*curv**0.5) + eps
#     )

#     return torch.acos(torch.clamp(cos_angle, min=-1+eps, max=1-eps))


def oxy_angle_stable(
    x: Tensor, 
    y: Tensor, 
    curv: float | Tensor = 1.0, 
    eps: float = 1e-8,
    debug: bool = False
) -> Tensor:
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle Oxy (O is the origin).

    Args:
        x: (B, D) Tensor
        y: (B, D) Tensor
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: small constant for numerical stability.
        debug: if True, print intermediate statistics.

    Returns:
        angle: (B,) Tensor with values in (0, pi).
    """

    # 1) 时间分量（加 clamp 防止 sqrt 负数）
    x_sq = torch.sum(x**2, dim=-1)
    y_sq = torch.sum(y**2, dim=-1)
    x_time = torch.sqrt(torch.clamp(1 / curv + x_sq, min=eps))
    y_time = torch.sqrt(torch.clamp(1 / curv + y_sq, min=eps))

    # 2) Lorentz 内积
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # 3) 分母
    denom_arg = torch.clamp(c_xyl**2 - 1, min=eps)  # 防止负数
    acos_denom = torch.sqrt(denom_arg)

    # 4) 分子
    acos_numer = y_time + c_xyl * x_time
    norm_x = torch.norm(x, dim=-1)

    # 5) 计算输入
    acos_input = acos_numer / (norm_x * acos_denom + eps)

    # 6) 截断输入到 [-1,1]
    acos_input = torch.clamp(acos_input, -1 + eps, 1 - eps)

    # 7) 反余弦
    angle = torch.acos(acos_input)

    if debug:
        if torch.isnan(angle).any() or torch.isinf(angle).any():
            print("[oxy_angle_stable] NaN/Inf detected!")
            print("x_time range:", x_time.min().item(), x_time.max().item())
            print("y_time range:", y_time.min().item(), y_time.max().item())
            print("c_xyl range:", c_xyl.min().item(), c_xyl.max().item())
            print("acos_denom range:", acos_denom.min().item(), acos_denom.max().item())
            print("acos_input range:", acos_input.min().item(), acos_input.max().item())

    return angle

# def oxy_angle_safe(
#     x: Tensor,
#     y: Tensor,
#     curv: float | Tensor = 1.0,
#     eps: float = 1e-8,
#     grad_clip: float = 10.0,  # 限制梯度大小
#     debug: bool = False
# ) -> Tensor:
#     """
#     Numerically stable + gradient-safe version of oxy_angle.

#     Args:
#         x: (B, D) Tensor
#         y: (B, D) Tensor
#         curv: positive scalar for negative hyperboloid curvature
#         eps: small constant for numerical stability
#         grad_clip: max allowed gradient magnitude for acos input
#         debug: if True, print debug info

#     Returns:
#         angle: (B,) Tensor in (0, pi)
#     """

#     # 1) 时间分量
#     x_sq = torch.sum(x**2, dim=-1)
#     y_sq = torch.sum(y**2, dim=-1)
#     x_time = torch.sqrt(torch.clamp(1 / curv + x_sq, min=eps))
#     y_time = torch.sqrt(torch.clamp(1 / curv + y_sq, min=eps))

#     # 2) Lorentz 内积
#     c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

#     # 3) 分母
#     denom_arg = torch.clamp(c_xyl**2 - 1, min=eps)
#     acos_denom = torch.sqrt(denom_arg)

#     # 4) 分子
#     acos_numer = y_time + c_xyl * x_time
#     norm_x = torch.norm(x, dim=-1)

#     # 5) 计算 acos_input
#     acos_input = acos_numer / (norm_x * acos_denom + eps)

#     # --- forward clamp ---
#     acos_input = torch.clamp(acos_input, -1 + eps, 1 - eps)

#     # --- backward gradient clip ---
#     acos_input_safe = acos_input.detach() + (acos_input - acos_input.detach()).clamp(-grad_clip, grad_clip)

#     # 6) angle
#     angle = torch.acos(acos_input_safe)

#     if debug:
#         if torch.isnan(angle).any() or torch.isinf(angle).any():
#             print("[oxy_angle_safe] NaN/Inf detected!")
#             print("acos_input range:", acos_input.min().item(), acos_input.max().item())
#             print("c_xyl range:", c_xyl.min().item(), c_xyl.max().item())

#     return angle

def oxy_angle_safe(
    x: Tensor,
    y: Tensor,
    curv: float | Tensor = 1.0,
    eps: float = 1e-8,
    grad_clip: float = 10.0,  # 限制梯度大小
    debug: bool = False
) -> Tensor:
    """
    Numerically stable + gradient-safe version of oxy_angle.

    Args:
        x: (B, D) Tensor
        y: (B, D) Tensor
        curv: positive scalar for negative hyperboloid curvature
        eps: small constant for numerical stability
        grad_clip: max allowed gradient magnitude for acos input
        debug: if True, print debug info

    Returns:
        angle: (B,) Tensor in (0, pi)
    """
    # --- 1) 时间分量 ---
    x_sq = torch.sum(x**2, dim=-1)
    y_sq = torch.sum(y**2, dim=-1)
    x_time = torch.sqrt(torch.clamp(1.0 / curv + x_sq, min=eps))
    y_time = torch.sqrt(torch.clamp(1.0 / curv + y_sq, min=eps))

    # --- 2) Lorentz 内积 ---
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # --- 3) 分母 ---
    denom_arg = torch.clamp(c_xyl**2 - 1.0, min=eps)
    acos_denom = torch.sqrt(denom_arg)

    # --- 4) 分子 ---
    acos_numer = y_time + c_xyl * x_time
    norm_x = torch.norm(x, dim=-1).clamp_min(eps)

    # --- 5) 计算 acos_input ---
    acos_input = acos_numer / (norm_x * acos_denom + eps)

    # --- forward clamp ---
    # 保证输入落在 (-1+eps, 1-eps) 内
    acos_input = torch.clamp(acos_input, -1.0 + eps, 1.0 - eps)

    # --- backward gradient clip ---
    # 保证梯度不会爆炸
    acos_input_safe = (
        acos_input.detach()
        + (acos_input - acos_input.detach()).clamp(-grad_clip, grad_clip)
    )

    # --- 6) angle ---
    angle = torch.acos(acos_input_safe)

    # --- Debug Info ---
    if debug:
        if torch.isnan(angle).any() or torch.isinf(angle).any():
            print("[oxy_angle_safe] NaN/Inf detected!")
            print("  acos_input range:", acos_input.min().item(), acos_input.max().item())
            print("  c_xyl range:", c_xyl.min().item(), c_xyl.max().item())
            print("  norm_x min:", norm_x.min().item())
            print("  denom_arg min:", denom_arg.min().item())

    return angle

def oxy_angle_eval(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))

    logger.info(f"x_time shape: {x_time.size()}")
    logger.info(f"y_time shape: {y_time.size()}")

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).

    # c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
    c_xyl = curv * (y @ x.T - y_time @ x_time.T)
    logger.info(f"c_xyl shape: {c_xyl.size()}")

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time.T
    logger.info(f"acos_numer shape: {acos_numer.size()}")
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    logger.info(f"acos_denom shape: {acos_denom.size()}")

    acos_input = acos_numer / (torch.norm(x, dim=-1, keepdim=True).T * acos_denom + eps)
    _angle = - torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle

# ----------  NEW: Minkowski inner product for two batches  ----------
def minkowski_dot(
    x: Tensor,                # space part, shape (..., D)
    y: Tensor,                # space part, shape (..., D)
    curv: float | Tensor = 1.0,
) -> Tensor:
    """
    Lorentzian inner product <x̄, ȳ>_L for vectors on the hyperboloid.

    Both x, y are **space components only** (time coord will be computed):
        t_x = sqrt(1/κ + ||x||²)
        t_y = sqrt(1/κ + ||y||²)

    Return shape: broadcasted (...), i.e. same batch dims as x/y (no extra axis).
    """
    # make sure x,y broadcastable on batch dims
    tx = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1))
    ty = torch.sqrt(1.0 / curv + torch.sum(y**2, dim=-1))
    # Minkowski: -t_x t_y + x·y
    return (x * y).sum(dim=-1) - tx * ty

# def oxy_cos_safe(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     curv: float | torch.Tensor = 1.0,
#     eps: float = 1e-8,
#     grad_clip: float = 10.0,
#     debug: bool = False,
# ) -> torch.Tensor:
#     """
#     Numerically stable + gradient-safe version that directly returns cos(angle).
#     Essentially returns the acos_input from oxy_angle_safe.
    
#     Returns:
#         cos_angle: (B,) Tensor in (-1, 1)
#     """
#     # --- 1) 时间分量 ---
#     x_sq = torch.sum(x**2, dim=-1)
#     y_sq = torch.sum(y**2, dim=-1)
#     x_time = torch.sqrt(torch.clamp(1.0 / curv + x_sq, min=eps))
#     y_time = torch.sqrt(torch.clamp(1.0 / curv + y_sq, min=eps))

#     # --- 2) Lorentz 内积 ---
#     c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

#     # --- 3) 分母 ---
#     denom_arg = torch.clamp(c_xyl**2 - 1.0, min=eps)
#     acos_denom = torch.sqrt(denom_arg)

#     # --- 4) 分子 ---
#     acos_numer = y_time + c_xyl * x_time
#     norm_x = torch.norm(x, dim=-1).clamp_min(eps)

#     # --- 5) cos(angle) = acos_input ---
#     acos_input = acos_numer / (norm_x * acos_denom + eps)
#     acos_input = torch.clamp(acos_input, -1.0 + eps, 1.0 - eps)

#     # --- backward gradient clip ---
#     cos_angle_safe = (
#         acos_input.detach()
#         + (acos_input - acos_input.detach()).clamp(-grad_clip, grad_clip)
#     )

#     if debug:
#         if torch.isnan(cos_angle_safe).any() or torch.isinf(cos_angle_safe).any():
#             print("[oxy_cos_safe] NaN/Inf detected!")
#             print("  acos_input range:", acos_input.min().item(), acos_input.max().item())
#             print("  c_xyl range:", c_xyl.min().item(), c_xyl.max().item())
#             print("  norm_x min:", norm_x.min().item())
#             print("  denom_arg min:", denom_arg.min().item())

#     return cos_angle_safe

def oxy_cos_safe(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: float | torch.Tensor = 1.0,
    eps: float = 1e-8,
    grad_clip: float = 10.0,
    debug: bool = False,
) -> torch.Tensor:
    x_sq = torch.sum(x**2, dim=-1)
    y_sq = torch.sum(y**2, dim=-1)
    x_time = torch.sqrt(torch.clamp(1.0 / curv + x_sq, min=eps))
    y_time = torch.sqrt(torch.clamp(1.0 / curv + y_sq, min=eps))

    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
    denom_arg = torch.clamp(c_xyl**2 - 1.0, min=eps)
    acos_denom = torch.sqrt(denom_arg)

    acos_numer = y_time + c_xyl * x_time
    norm_x = torch.norm(x, dim=-1).clamp_min(eps)

    acos_input = acos_numer / (norm_x * acos_denom + eps)
    acos_input = torch.clamp(acos_input, -1.0 + eps, 1.0 - eps)

    # ---- 替换梯度安全实现 ----
    # 原版 detach 导致梯度断，这里改成 softclip
    cos_angle_safe = grad_clip * torch.tanh(acos_input / grad_clip)
  
    if debug:
        if torch.isnan(cos_angle_safe).any() or torch.isinf(cos_angle_safe).any():
            print("[oxy_cos_safe] NaN/Inf detected!")
            print("  acos_input range:", acos_input.min().item(), acos_input.max().item())
            print("  c_xyl range:", c_xyl.min().item(), c_xyl.max().item())
            print("  norm_x min:", norm_x.min().item())
            print("  denom_arg min:", denom_arg.min().item())

    return cos_angle_safe

def oxy_tangent_cos(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-9):
    """
    Cosine of the angle between geodesics Ox and Oy at the hyperboloid origin.
    Equivalent to cosine of angle between tangent vectors at O.
    """
    # Spatial components are tangent vectors at the origin
    x_norm = torch.norm(x, dim=-1)
    y_norm = torch.norm(y, dim=-1)

    dot = torch.sum(x * y, dim=-1)

    cos_angle = dot / (x_norm * y_norm + eps)
    return torch.clamp(cos_angle, -1.0, 1.0)
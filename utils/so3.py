import torch
import math

def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of vectors (N x 3) into skew-symmetric matrices (N x 3 x 3)
    """
    zeros = torch.zeros_like(v[:, 0])
    h = torch.stack([
        zeros, -v[:, 2], v[:, 1],
        v[:, 2], zeros, -v[:, 0],
        -v[:, 1], v[:, 0], zeros
    ], dim=1).reshape(-1, 3, 3)
    return h

def exp_so3(skew: torch.Tensor) -> torch.Tensor:
    """
    Computes the matrix exponential of a batch of skew-symmetric matrices (N x 3 x 3)
    using the Rodrigues' formula for SO(3)
    """
    theta = torch.norm(skew[:, [2, 0, 1], [1, 2, 0]], dim=1).unsqueeze(1).unsqueeze(2)  # shape (N,1,1)
    I = torch.eye(3, device=skew.device).unsqueeze(0).expand_as(skew)
    theta = theta + 1e-8

    A = torch.sin(theta) / theta
    B = (1 - torch.cos(theta)) / (theta ** 2)
    
    return I + A * skew + B * torch.bmm(skew, skew)

def f_igso3_small(omega: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Approximates the density function of omega for the isotropic Gaussian distribution on SO(3).
    """
    eps = (sigma / math.sqrt(2)) ** 2
    pi = math.pi
    small_number = 1e-9
    small_num = small_number / 2 
    small_dnm = (1 - torch.exp(-pi**2 / eps) * (2 - 4 * (pi**2) / eps)) * small_number

    return (0.5 * math.sqrt(pi) * (eps ** -1.5) * 
            torch.exp((eps - (omega**2 / eps))/4) / (torch.sin(omega/2) + small_num) *
            (small_dnm + omega - ((omega - 2*pi)*torch.exp(pi * (omega - pi) / eps) 
                                   + (omega + 2*pi)*torch.exp(-pi * (omega+pi) / eps))))

def angle_density_unif(omega: torch.Tensor) -> torch.Tensor:
    """
    Marginal density of rotation angle for uniform density on SO(3)
    """
    return (1 - torch.cos(omega)) / math.pi

def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    One-dimensional linear interpolation for monotonically increasing sample points.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1]) 
    b = fp[:-1] - (m * xp[:-1])                

    indices = torch.sum(torch.ge(x[:, None], xp[None, :]), dim=1) - 1 
    indices = torch.clamp(indices, 0, len(m) - 1)

    return m[indices] * x + b[indices]

def sample_normal_SO3(batch_size: int = 1,
                      horizon: int = 1,
                      mean: torch.Tensor = None,
                      std: float = 1.0) -> torch.Tensor:
    """
    Samples from the Isotropic Gaussian distribution on SO(3) (IGSO(3)).

    Parameters:
        batch_size (int): Number of samples in the batch.
        horizon (int): Number of samples per batch element.
        mean (torch.Tensor): Mean rotation matrix of shape (3, 3). If None, identity is used.
        std (float): Standard deviation of the distribution.

    Returns:
        torch.Tensor: Samples of shape (batch_size, horizon, 3, 3)
    """
    device = mean.device if mean is not None else 'cpu'
    total_samples = batch_size

    num_omegas = 1024
    omega_grid = torch.linspace(0, math.pi, num_omegas + 1, device=device)[1:]  # skip omega=0
    pdf = f_igso3_small(omega_grid, std) * angle_density_unif(omega_grid)
    dx = omega_grid[1] - omega_grid[0]
    cdf = torch.cumsum(pdf, dim=-1) * dx
    cdf = cdf / cdf[-1] 

    rand_vals = torch.rand(total_samples, device=device)
    omegas = interp(rand_vals, cdf, omega_grid)

    axes = torch.randn(total_samples, 3, device=device)
    axes = axes / torch.norm(axes, dim=1, keepdim=True)

    axis_angles = omegas.unsqueeze(1) * axes

    skew_matrices = hat(axis_angles)
    rotations = exp_so3(skew_matrices)

    if mean is not None:
        rotations = torch.matmul(mean.unsqueeze(0), rotations)

    rotations = rotations.view(batch_size, 3, 3)
    rotations = rotations.unsqueeze(1).repeat(1, horizon, 1, 1)
    return rotations

def procrustes_to_so3(matrix_9d: torch.Tensor) -> torch.Tensor:
    """
    Projects a batched 9D tensor (flattened 3x3 matrices) to SO(3) using SVD.
    
    Args:
        matrix_9d: Input tensor of shape [..., 9]
    
    Returns:
        Rotation matrices of shape [..., 3, 3]
    """
    M = matrix_9d.view(*matrix_9d.shape[:-1], 3, 3)
    
    U, S, Vt = torch.linalg.svd(M)
    
    det = torch.det(U @ Vt)
    
    S_adj = torch.zeros_like(U)
    S_adj[..., 0, 0] = 1.0
    S_adj[..., 1, 1] = 1.0
    S_adj[..., 2, 2] = torch.where(det < 0, -1.0, 1.0)
    
    R = U @ S_adj @ Vt
    return R

def zhou_6d_to_so3(vector_6d: torch.Tensor) -> torch.Tensor:
    """
    Converts a batched 6D tensor to SO(3) using Gram-Schmidt and cross product.
    
    Args:
        vector_6d: Input tensor of shape [..., 6]
    
    Returns:
        Rotation matrices of shape [..., 3, 3]
    """
    a = vector_6d[..., :3]
    b = vector_6d[..., 3:6]

    u1 = a / torch.norm(a, dim=-1, keepdim=True).clamp(min=1e-12)
    
    # Project b onto u1 and orthogonalize
    b_proj = (torch.sum(b * u1, dim=-1, keepdim=True) * u1)
    u2 = b - b_proj

    u2 = u2 / torch.norm(u2, dim=-1, keepdim=True).clamp(min=1e-12)

    u3 = torch.cross(u1, u2, dim=-1)

    return torch.stack([u1, u2, u3], dim=-1)

def BM_2_R6(bm: torch.Tensor) -> torch.Tensor:
    """
    Converts batched rotation matrices (3x3) to Zhou's 6D representation.
    
    Args:
        bm: Input tensor of shape [..., 3, 3] (batch of rotation matrices)
    
    Returns:
        Tensor of shape [..., 6] (Zhou's 6D representation)
    """
    original_shape = bm.shape[:-2]
    
    return bm.transpose(-2, -1).reshape(*original_shape, -1)[...,:6]

def BM_2_R9(bm: torch.Tensor) -> torch.Tensor:
    """
    Converts batched rotation matrices (3x3) to 9D representation.
    
    Args:
        bm: Input tensor of shape [..., 3, 3] (batch of rotation matrices)
    
    Returns:
        Tensor of shape [..., 9]
    """
    original_shape = bm.shape[:-2]
    return bm.view(*original_shape, 9)

if __name__=='__main__':
    sample = sample_normal_SO3(1200, 3, None, torch.tensor(.2))
    print(sample[:5])
    print(sample[0][0] @ sample[0][0].T)

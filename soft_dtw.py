import torch


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.norm(x - y, p=2)

def spherical_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    cos_theta = torch.clamp(
        torch.dot(x, y) / (torch.norm(x) * torch.norm(y)),
        -1.0 + eps, 1.0 - eps
    )
    return torch.arccos(cos_theta)

def dtw_distance(A: torch.Tensor, B: torch.Tensor, dist_func):
    n, m = A.size(0), B.size(0)
    dtw = torch.full((n + 1, m + 1), float('inf'), device=A.device)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_func(A[i - 1], B[j - 1])
            dtw[i, j] = cost + torch.min(torch.stack([
                dtw[i - 1, j],     # insertion
                dtw[i, j - 1],     # deletion
                dtw[i - 1, j - 1]  # match
            ]))

    return dtw[n, m]
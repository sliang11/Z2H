import torch
import numpy as np
from time import perf_counter
from typing import Union


def preprocess_examples(examples: torch.Tensor):
    if examples.ndim == 2:
        # 2D case: z-norm row-wise
        mean = examples.mean(dim=1, keepdim=True)
        std = examples.std(dim=1, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        out = (examples - mean) / std
        return out

    elif examples.ndim == 3:
        # 3D case: z-norm along last dimension
        mean = examples.mean(dim=2, keepdim=True)         # [n,h,1]
        std = examples.std(dim=2, keepdim=True)          # [n,h,1]
        std = torch.where(std == 0, torch.ones_like(std), std)
        z = (examples - mean) / std                       # [n,h,w]

        # flatten last 2 dims
        n, h, w = z.shape
        return z.reshape(n, h * w)                        # [n, h*w]

    else:
        raise ValueError("examples must be 2D or 3D")


def knn_self_similarity_blockwise(examples: torch.Tensor, k: int, block_size: int = 2048):


    device = examples.device
    examples = preprocess_examples(examples)
    n, d = examples.shape

    # Precompute L2 norms for all samples
    norms = (examples ** 2).sum(dim=1)  # [n]

    knn_dist_mat = torch.zeros((n, k), device=device)
    knn_inds_mat = torch.empty((n, k), dtype=torch.long, device=device)

    for start in range(0, n, block_size):
        tic = perf_counter()

        end = min(start + block_size, n)
        B = end - start

        q = examples[start:end]                  # [B, d]
        q_norm = norms[start:end].unsqueeze(1)   # [B, 1]

        # Compute squared L2 distances: ||q||^2 + ||x||^2 - 2 q·x
        dot = q @ examples.T                     # [B, n]
        dist = q_norm + norms.unsqueeze(0) - 2 * dot
        dist = torch.clamp(dist, min=0.0)

        # Exclude self-distance for rows inside this block
        row_ids = torch.arange(start, end, device=device)
        dist[torch.arange(B), row_ids] = 1e10

        # Select k nearest neighbors
        dist_k, inds_k = torch.topk(dist, k=k, largest=False)
        dist_k = torch.sqrt(dist_k)
        assert (dist_k >= 0).all()

        knn_dist_mat[start:end] = dist_k
        knn_inds_mat[start:end] = inds_k

        del q, q_norm, dot, dist, dist_k, inds_k
        torch.cuda.empty_cache()

        print(f'Block {start}-{end} time: {perf_counter() - tic}')


    assert (knn_dist_mat >= 0).all()
    return knn_dist_mat.cpu().numpy(), knn_inds_mat.cpu().numpy()


def calc_knn_from_ndarray(
        multi_channel_examples: np.ndarray, channels: Union[list, np.ndarray, int] = None,
        k: int = 1000, block_size: int = 20000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    examples = multi_channel_examples if channels is None else multi_channel_examples[:, channels, :]
    examples = torch.from_numpy(examples).float().to(device)
    knn_dist_mat, knn_inds_mat = knn_self_similarity_blockwise(
        examples, min(k, examples.shape[0]), block_size=block_size)
    return knn_dist_mat, knn_inds_mat


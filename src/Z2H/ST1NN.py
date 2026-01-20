import numpy as np
import torch
from time import perf_counter


def st1nn_no_sc_gpu(knn_dist_mat_: np.ndarray, knn_inds_mat_: np.ndarray,
                    init_p_inds_: np.ndarray, init_n_inds_: np.ndarray,
                    max_iters=100000, n_neighbors=10000, dist_inf=1e20,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), time_every=10000):

    init_tic = perf_counter()

    """
    Initialize on GPU
    """

    assert knn_dist_mat_.shape == knn_inds_mat_.shape
    n_rows, n_cols = knn_dist_mat_.shape
    n_neighbors = min(n_neighbors, n_cols)
    n_cols = n_neighbors
    knn_dist_mat = torch.from_numpy(knn_dist_mat_[:, :n_neighbors]).float().to(device)
    knn_inds_mat = torch.from_numpy(knn_inds_mat_)[:, :n_neighbors].int().to(device)
    assert knn_dist_mat.shape == knn_inds_mat.shape == (n_rows, n_cols)

    n_init_p, n_init_n = len(init_p_inds_), len(init_n_inds_)
    ranked_inds = torch.empty(n_init_p + max_iters, dtype=torch.int32, device=device)
    ranked_inds[:n_init_p] = torch.from_numpy(init_p_inds_).to(device=device)

    nn_in_p = torch.empty(max_iters, dtype=torch.int32, device=device)
    pu_dists = torch.empty(max_iters, dtype=torch.float32, device=device)

    """
    Handle initially labeled examples
    """

    init_u_inds_ = np.array([i for i in range(n_rows) if i not in np.concatenate([init_p_inds_, init_n_inds_])])
    init_p_inds = torch.from_numpy(init_p_inds_).long().to(device=device)
    init_n_inds = torch.from_numpy(init_n_inds_).long().to(device=device)
    init_u_inds = torch.from_numpy(init_u_inds_).long().to(device=device)

    dist_ub = torch.max(knn_dist_mat) + 1e-5

    # N -> all: invalid forever
    knn_dist_mat[init_n_inds, :] = dist_inf

    # PP, PN, UP, UN: invalid forever
    pu_inds = torch.cat((init_p_inds, init_u_inds))
    init_pn_inds = torch.cat((init_p_inds, init_n_inds))
    invalid_f = torch.where(torch.isin(knn_inds_mat[pu_inds], init_pn_inds))
    knn_dist_mat[pu_inds[invalid_f[0]], invalid_f[1]] = dist_inf

    # UU: invalid temporarily
    invalid_t = torch.where(torch.isin(knn_inds_mat[init_u_inds], init_u_inds))
    knn_dist_mat[init_u_inds[invalid_t[0]], invalid_t[1]] += dist_ub


    print(f'Initialization time: {perf_counter() - init_tic} seconds')

    iter_tic = tic = perf_counter()
    n_iters = max_iters
    for i_iter in range(max_iters):

        idx = torch.argmin(knn_dist_mat)
        row, col = idx // n_cols, idx % n_cols
        pu_dist = knn_dist_mat[row, col].item()

        if pu_dist == dist_inf:
            n_iters = i_iter
            break

        assert pu_dist < dist_ub
        next_nn_in_p, next_p = row.item(), knn_inds_mat[row, col].item()
        invalid_f = torch.where(knn_inds_mat[pu_inds] == next_p)
        assert len(invalid_f) == 2 and len(invalid_f[0]) == len(invalid_f[1])
        if len(invalid_f[0]) > 0:
            knn_dist_mat[pu_inds[invalid_f[0]], invalid_f[1]] = dist_inf

        # UU -> PU
        knn_dist_mat[next_p, (knn_dist_mat[next_p] != dist_inf) & (knn_dist_mat[next_p] >= dist_ub)] -= dist_ub

        assert next_nn_in_p in ranked_inds[:n_init_p + i_iter]
        assert next_p not in ranked_inds[:n_init_p + i_iter]
        assert next_p not in nn_in_p[:i_iter]

        nn_in_p[i_iter] = next_nn_in_p
        pu_dists[i_iter] = pu_dist


        ranked_inds[n_init_p + i_iter] = next_p

        if (i_iter + 1) % time_every == 0:
            print(f'Iterations {i_iter + 2 - time_every}-{i_iter + 1}: time = {perf_counter() - tic} seconds.')
            tic = perf_counter()

    ranked_inds = ranked_inds[n_init_p: n_init_p + n_iters].cpu().numpy()
    nn_in_p = nn_in_p[:n_iters].cpu().numpy()
    pu_dists = pu_dists[:n_iters].cpu().numpy()

    total_time = perf_counter() - iter_tic
    print(f'Iterations ended at {n_iters}: total time = {total_time} seconds.')

    return ranked_inds, nn_in_p, pu_dists, total_time
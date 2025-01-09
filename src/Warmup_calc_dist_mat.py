#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
main_dir = os.path.dirname(__file__)

import sys
sys.path.append(main_dir)

from util.file_util import FileReader, FileWriter, DirProcessor
from util.calc_util import z_norm_2d_by_row
import numpy as np
import os
import argparse
from time import perf_counter
from numba import njit


class KNN:
    def __init__(self, k=1000):
        self.k = k

    @staticmethod
    @njit
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    @staticmethod
    # @njit
    def euclidean_distance_ts_to_set(X1, x2):  # 这个在njit下反而比上面那个更慢，为什么？？？
        return np.sqrt(np.sum((X1 - x2) ** 2, axis=1))

    # @njit
    def knn_queries(self, queries, data):
        n_queries = queries.shape[0]
        top_k_distances = np.zeros((n_queries, self.k), dtype=np.float32)
        top_k_indices = np.zeros((n_queries, self.k), dtype=int)

        for i, query in enumerate(queries):
            distances = np.array([self.euclidean_distance(query, x) for x in data])
            sorted_indices = np.argsort(distances)
            assert distances[sorted_indices[0]] == 0
            k_indices = sorted_indices[1: self.k + 1]  # avoid having the example itself as the NN

            top_k_distances[i] = distances[k_indices]
            top_k_indices[i] = k_indices

        return top_k_distances, top_k_indices


def concat_cached_files(save_path, data_id, n_examples, n_neighbors, n_examples_per_chunk):

    print('Concatenating all cache files for final results...', end='')

    for mat_id, mat_dtype in zip(("dists", "inds"), (np.float32, int)):
        mat = np.empty((n_examples, n_neighbors), dtype=mat_dtype)

        for start in range(0, n_examples, n_examples_per_chunk):
            finish = min(start + n_examples_per_chunk, n_examples)
            full_cache_fname = os.path.join(save_path, f"cache_knn_{mat_id}_{data_id}_{start}-{finish}.pkl")
            mat[start: finish] = FileReader.load_pickle(full_cache_fname)

        full_final_fname = os.path.join(save_path, f"knn_{mat_id}_{data_id}.pkl")
        FileWriter.dump_pickle(mat, full_final_fname)

    # remove all the cache files
    for mat_id, mat_dtype in zip(("dists", "inds"), (np.float32, int)):
        for start in range(0, n_examples, n_examples_per_chunk):
            finish = min(start + n_examples_per_chunk, n_examples)
            full_cache_fname = os.path.join(save_path, f"cache_knn_{mat_id}_{data_id}_{start}-{finish}.pkl")
            os.remove(full_cache_fname)
    os.remove(os.path.join(save_path, f"next_query_ind_{data_id}.pkl"))

    print('All done!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('data_id', choices=('MO1', 'MO2', 'MO3',), help='Data ID.')
    parser.add_argument('--n_neighbors', type=int, default=1000, help='Number of nearest neighbors for kNN.')
    parser.add_argument('--data_path', default=f'{main_dir}/data', help='Path to input dataset.')
    parser.add_argument('--dist_mat_path', default=f'{main_dir}/distances',
                        help='Path to pre-computed distance matrices')
    parser.add_argument('--n_examples_per_chunk', type=int, default=500, help='Number of examples in each cache file.')

    args = parser.parse_args()
    data_id = args.data_id
    n_neighbors = args.n_neighbors
    data_path = args.data_path
    save_path = args.dist_mat_path
    n_examples_per_chunk = args.n_examples_per_chunk

    print(f"\n\n======= Warmup: calculate k-NN distance matrix for Mouse {data_id} =======\n\n")

    DirProcessor.create_dir(save_path)

    final_dist_fname = f"knn_dists_{data_id}.pkl"
    final_ind_fname = f"knn_inds_{data_id}.pkl"
    next_query_ind_fname = f"next_query_ind_{data_id}.pkl"

    full_fdf = os.path.join(save_path, final_dist_fname)
    full_fif = os.path.join(save_path, final_ind_fname)
    full_nqif = os.path.join(save_path, next_query_ind_fname)

    # load data
    data_fname = f"{data_id}.pkl"
    full_data_fname = os.path.join(data_path, data_fname)

    obj = FileReader.load_pickle(full_data_fname)
    # exec(gen_cmd_print_variables('full_data_fname, obj'))

    train_samples, val_samples, test_samples, train_bi_labels, val_bi_labels, test_bi_labels = \
        FileReader.load_pickle(full_data_fname)
    data = np.vstack((train_samples, val_samples))
    data = z_norm_2d_by_row(data)
    n_examples = data.shape[0]
    # exec(gen_cmd_print_variables('n_examples'))

    # start from the last checkpoint
    if os.path.exists(full_fdf) and os.path.exists(full_fif):
        print("Already done!")
        exit(0)
    if not os.path.exists(full_nqif):
        next_start = 0
    else:
        next_start = FileReader.load_pickle(full_nqif)

    # if all that's left to do is concatenate the cache files
    if next_start == n_examples:
        concat_cached_files(save_path, data_id, n_examples, n_neighbors, n_examples_per_chunk)
        exit(0)

    # initialize KNN
    knn = KNN(k=n_neighbors)

    # for chunks of queries
    for start in range(next_start, n_examples, n_examples_per_chunk):
        finish = min(start + n_examples_per_chunk, n_examples)
        print(f"Querying {start + 1} - {finish} / {n_examples}...", end="")

        tic = perf_counter()
        queries = data[start: finish]

        # do the knn queries
        knn_dists, knn_inds = knn.knn_queries(queries, data)
        assert knn_dists.shape == (finish - start, n_neighbors)

        # save to cache files
        cache_knn_dist_fname = f"cache_knn_dists_{data_id}_{start}-{finish}.pkl"
        full_ckdf = os.path.join(save_path, cache_knn_dist_fname)
        FileWriter.dump_pickle(knn_dists, full_ckdf)

        cache_knn_inds_fname = f"cache_knn_inds_{data_id}_{start}-{finish}.pkl"
        full_ckif = os.path.join(save_path, cache_knn_inds_fname)
        FileWriter.dump_pickle(knn_inds, full_ckif)

        FileWriter.dump_pickle(finish, full_nqif)
        toc = perf_counter()
        print(f"Done in {toc - tic}s.")

    del data, queries, knn
    concat_cached_files(save_path, data_id, n_examples, n_neighbors, n_examples_per_chunk)



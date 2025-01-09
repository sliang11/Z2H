#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
main_dir = os.path.dirname(__file__)

import sys
sys.path.append(main_dir)

import numpy as np
from util.file_util import FileReader, FileWriter, DirProcessor
from deepod.models.tabular import DeepSVDD, DeepIsolationForest, ICL
import argparse


def decide_query_order(
        n_examples, strategy, knn_dist_mat=None, knn_inds_mat=None, amplitudes=None, rand_seed=0
):

    data_driven_strategy_to_method = {
        'DeepSVDD': DeepSVDD,
        'ICL': ICL,
        'DIF': DeepIsolationForest,
    }
    data_driven_strategies = list(data_driven_strategy_to_method.keys())

    supported_strategies = ["random", "amp", "amp_dist",
                            "dist", "dist_amp", ] + data_driven_strategies
    flg = False
    for strategy_ in supported_strategies:
        if strategy.startswith(strategy_):
            flg = True
            break
    if not flg:
        raise Exception("Strategy not supported!")

    # determine the queries order
    if strategy in data_driven_strategies:
        method = data_driven_strategy_to_method[strategy]
        clf = method(verbose=0)
        clf.fit(examples)
        anomaly_scores = clf.decision_function(examples)
        order = np.argsort(-anomaly_scores)
        return order

    if strategy.startswith("amp"):
        order = np.argsort(-amplitudes)  # descending

        if strategy == "amp_dist":  # use the median 1NN distance as the filter
            amp_ordered_1nn_dists = knn_dist_mat[order, 0]
            dist_th = np.median(amp_ordered_1nn_dists)
            invalid = np.where(amp_ordered_1nn_dists > dist_th)[0]
            valid = np.delete(np.arange(n_examples), invalid)
            # print(f"Filtered {len(invalid)} / {n_examples} examples.")
            if 0 < len(valid) < n_examples:
                order = np.concatenate((order[valid], order[invalid]))

        return order

    if strategy.startswith("dist"):

        nn_dists = knn_dist_mat[:, 0]
        nn_inds = knn_inds_mat[:, 0]

        dist_ordered_inds = np.argsort(nn_dists)
        dist_ordered_nn_inds = nn_inds[dist_ordered_inds]
        if strategy == "dist_amp":
            amp_th = np.median(amplitudes)

            # do filtering for both the example itself and its nearest neighbor
            invalid = np.where(amplitudes[dist_ordered_inds] < amp_th)[0]
            invalid_ = np.where(amplitudes[dist_ordered_nn_inds] < amp_th)[0]
            invalid = np.union1d(invalid, invalid_)
            valid = np.delete(np.arange(len(dist_ordered_inds)), invalid)

            if 0 < len(valid) < len(dist_ordered_inds):
                dist_ordered_inds = np.concatenate((dist_ordered_inds[valid], dist_ordered_inds[invalid]))
                dist_ordered_nn_inds = np.concatenate(
                    (dist_ordered_nn_inds[valid], dist_ordered_nn_inds[invalid]))

        # queries in pairs (the example itself, and its NN); use amplitude-descending order in each pair
        chosen = np.zeros(n_examples, dtype=np.int8)
        order = []
        for ind, nn_ind in zip(dist_ordered_inds, dist_ordered_nn_inds):
            amp_ind, amp_nn_ind = amplitudes[ind], amplitudes[nn_ind]
            inds = (ind, nn_ind) if amp_ind > amp_nn_ind else (nn_ind, ind)
            for ind in inds:
                if not chosen[ind]:
                    order.append(ind)
                    chosen[ind] = 1
        order = np.array(order)
        return order

    # random
    np.random.seed(rand_seed)
    return np.random.permutation(n_examples)


def query_by_round_robin(n_examples, strategies, real_labels, max_n_queries=None, knn_dist_mat=None, knn_inds_mat=None,
                         amplitudes=None, rand_seed=0):

    if max_n_queries is None:
        max_n_queries = n_examples

    orders_by_strategy = [
        decide_query_order(
            n_examples, strategy, knn_dist_mat=knn_dist_mat, knn_inds_mat=knn_inds_mat,
            amplitudes=amplitudes, rand_seed=rand_seed
        ) for strategy in strategies
    ]

    # n_strategies = len(strategies)
    # next_to_query_by_strategy = np.empty(n_strategies, dtype=int)  # NOTE: this corresponds to the indices in "order", not in examples
    # next_to_query_by_strategy[0] = 0
    queried = np.zeros(n_examples, dtype=np.int8)
    queried_inds, queried_labels = [], []
    while len(queried_inds) != max_n_queries:

        # determine the examples to queries in this round
        inds_to_query = []
        for i, order in enumerate(orders_by_strategy):
            for j, next in enumerate(order):
                if not (queried[next] or next in inds_to_query):
                    inds_to_query.append(next)
                    orders_by_strategy[i] = order[j + 1:]
                    break
            if len(inds_to_query) == max_n_queries - len(queried_inds):  # early stopping for the last round
                break

        # do the querying
        queried_inds += inds_to_query
        # exec(gen_cmd_print_variables("inds_to_query"))
        inds_to_query = np.array(inds_to_query)
        queried_labels += list(real_labels[inds_to_query])
        queried[inds_to_query] = 1

    return np.array(queried_inds), np.array(queried_labels)


def evaluate(queried_labels, p_label=1, max_n_queries=100):
    if max_n_queries is None:
        max_n_queries = len(queried_labels)

    cnt_p, n_queries_first_p = 0, -1
    precisions_by_n_queries = np.empty(max_n_queries)

    for i, label in enumerate(queried_labels):
        if label == p_label:
            cnt_p += 1
            if n_queries_first_p == -1:
                n_queries_first_p = i + 1  # NOTE: +1
        precisions_by_n_queries[i] = cnt_p / (i + 1)
    return n_queries_first_p, precisions_by_n_queries


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('data_id', choices=('MO1', 'MO2', 'MO3',), help='Data ID.')
    parser.add_argument('--data_path', default=f'{main_dir}/data', help='Path to input dataset.')
    parser.add_argument('--dist_mat_path', default=f'{main_dir}/distances', help='Path to pre-computed distance matrices')
    parser.add_argument('--result_path', default=f'{main_dir}/result', help='Result output path.')

    args = parser.parse_args()
    data_id = args.data_id
    data_path = args.data_path
    dist_mat_path = args.dist_mat_path
    save_path = args.result_path

    n_rand_seeds = 10


    print(f"\n\n======= Module 1: active initialization for Mouse {data_id} =======\n\n")
    DirProcessor.create_dir(save_path)

    rand_strategy_combos = [(f"random_{rand_seed}",) for rand_seed in range(n_rand_seeds)]
    non_rand_strategy_combos = [
        ("amp",),
        ("amp_dist",),
        ("dist",),
        ("dist_amp",),
        ("DeepSVDD", ),
        ("ICL", ),
        ("DIF", ),
    ]
    strategy_combos = rand_strategy_combos + non_rand_strategy_combos


    # load the data
    print("Loading data...", end="")
    fname = f"{data_id}.pkl"
    full_fname = os.path.join(data_path, fname)
    train_examples, val_examples, test_examples, train_bi_labels, val_bi_labels, test_bi_labels = \
        FileReader.load_pickle(full_fname)
    real_labels = np.concatenate((train_bi_labels, val_bi_labels))
    n_p = len(np.where(real_labels == 1)[0])
    print("Done.")

    # get amplitudes
    examples = np.concatenate((train_examples, val_examples))
    amplitudes = np.max(examples, axis=1) - np.min(examples, axis=1)
    n_examples = len(amplitudes)
    print(f"There are {n_p} positive examples among all {n_examples} examples.")
    del train_examples, val_examples, test_examples, train_bi_labels, val_bi_labels, test_bi_labels

    # load the knn matrices
    print("Loading KNN distances...", end="")
    knn_dist_fname = os.path.join(dist_mat_path, f"knn_dists_{data_id}.pkl")
    knn_dist_mat = FileReader.load_pickle(knn_dist_fname)
    knn_inds_fname = os.path.join(dist_mat_path, f"knn_inds_{data_id}.pkl")
    knn_inds_mat = FileReader.load_pickle(knn_inds_fname)
    print("Done.\n")

    # conduct and evaluate active initiation
    for strategies in strategy_combos:

        assert len(strategies) == 1
        strategy = strategies[0]

        print(f"***** Current strategy: {strategy} *****")

        # Randomness in the current strategies?
        rand_seed = 0
        if strategy.startswith("random"):
            rand_seed = int(strategy.split("_")[-1])

        save_fname = os.path.join(save_path, f"Module_1_active_init_results_{data_id}_{strategy}.pkl")

        if os.path.exists(save_fname):
            queried_inds, queried_labels, n_queries_first_p, precisions_by_n_queries = \
                FileReader.load_pickle(save_fname)
        else:
            queried_inds, queried_labels = query_by_round_robin(
                n_examples, strategies, real_labels, max_n_queries=n_examples,
                knn_dist_mat=knn_dist_mat, knn_inds_mat=knn_inds_mat,
                amplitudes=amplitudes, rand_seed=rand_seed)
            n_queries_first_p, precisions_by_n_queries = evaluate(queried_labels, max_n_queries=n_examples)

            save_obj = queried_inds, queried_labels, n_queries_first_p, precisions_by_n_queries

            FileWriter.dump_pickle(save_obj, save_fname)

        print(f"First hit = {n_queries_first_p}, "
              f"p@10 = {precisions_by_n_queries[9]}, "
              f"p@20 = {precisions_by_n_queries[19]}, "
              f"p@50 = {precisions_by_n_queries[49]}\n"
              )

    print("All done!")











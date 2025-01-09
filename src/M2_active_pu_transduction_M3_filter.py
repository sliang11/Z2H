#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
main_dir = os.path.dirname(__file__)

import sys
sys.path.append(main_dir)
sys.setrecursionlimit(10000)    # To prevent "RunTimeError: maximum recursion depth exceeded" when deep-copying the ST-Tree

import numpy as np
import os
from time import perf_counter
from copy import deepcopy
from util.file_util import FileReader, FileWriter
from util.calc_util import unique_with_all_indices_1d
from util.evaluation_util import prf
import argparse
from util.self_training_tree import SelfTrainingTree as STTree
from typing import Union
from util.pu_stopping_criteria import PUStoppingCriteria as PUSC


def one_query(ind_to_query, budget, real_labels, timing=False, prev_tic_active=None):

    if timing:
        if prev_tic_active is not None:
            inter_active_query_time = perf_counter() - prev_tic_active
        else:
            inter_active_query_time = None

    q_label = real_labels[ind_to_query]
    budget -= 1

    if not timing:
        return q_label, budget

    tic_active = perf_counter()
    return q_label, budget, inter_active_query_time, tic_active


def update_variables_active_sc(tgt_chain_val,
                               sttree: STTree, chains, inf_by_ind, labels_by_ind,
                               i_rollback_in_chain=None,):

    # get one_inds and zero_inds, and update sttree
    if i_rollback_in_chain is None:
        one_inds = tgt_chain_val
        zero_inds = []
    else:
        one_inds = tgt_chain_val[:i_rollback_in_chain]
        rollback_ind = tgt_chain_val[i_rollback_in_chain]
        zero_inds = sttree.delete(ind=rollback_ind)

    # update chains and inf_by_ind
    inf_by_ind[tgt_chain_val] = int(1e10)
    del chains[tgt_chain_val[-1]]

    keys_to_del = []
    for key, val in chains.items():

        if tgt_chain_val[0] != val[0]:
            continue

        length = min(len(tgt_chain_val), len(val))
        diff = np.abs(tgt_chain_val[:length] - val[:length])
        l_common = len(np.where(diff == 0)[0])
        if i_rollback_in_chain is None or l_common <= i_rollback_in_chain:
            inf_by_ind[val[:l_common]] = int(1e10)
            chains[key] = val[l_common:]

        else:
            inf_by_ind[val] = int(1e10)
            keys_to_del.append(key)
    for key in keys_to_del:
        del chains[key]

    # update labels_by_ind
    if len(zero_inds) > 0:
        labels_by_ind[zero_inds] = 0
    if len(one_inds) > 0:
        labels_by_ind[one_inds] = 1

    return sttree, chains, inf_by_ind, labels_by_ind


def find_rollback(chain_val, budget, sttree:STTree, labels_by_ind, real_labels):

    start, finish = 0, len(chain_val) - 1
    queried_in_chain = {chain_val[-1]: 0}
    i_bsf = 0
    while start <= finish:
        i_query = (start + finish) // 2
        ind = chain_val[i_query]
        assert labels_by_ind[ind] == -1
        label = -1 if ind not in queried_in_chain.keys() else queried_in_chain[ind]

        if label == -1:
            if budget == 0:
                cur_queried_inds = list(queried_in_chain.keys())
                cur_queried_inds = [ind for ind in cur_queried_inds if ind != chain_val[-1]]
                return i_bsf, budget, cur_queried_inds
            label, budget = one_query(ind, budget, real_labels)
            queried_in_chain[ind] = label

        if label == 1:
            if i_bsf < i_query + 1:
                i_bsf = i_query + 1
            start = i_query + 1
            continue

        # label == 0，need to check the parent label
        parent_ind = sttree.get_parent_ind(ind)
        parent_label = labels_by_ind[parent_ind] if parent_ind not in queried_in_chain.keys() \
            else queried_in_chain[parent_ind]

        if parent_label == -1:
            if budget == 0:
                cur_queried_inds = list(queried_in_chain.keys())
                cur_queried_inds = [ind for ind in cur_queried_inds if ind != chain_val[-1]]
                return i_bsf, budget, cur_queried_inds
            parent_label, budget = one_query(parent_ind, budget, real_labels)
            queried_in_chain[parent_ind] = parent_label

        if parent_label == 1:
            cur_queried_inds = list(queried_in_chain.keys())
            cur_queried_inds = [ind for ind in cur_queried_inds if ind != chain_val[-1]]
            return i_query, budget, cur_queried_inds

        finish = i_query - 1


# The first step of ST-1NN, i.e. rank the U examples (without applying the SC)
def st1nn_no_sc(
        knn_dist_mat, knn_inds_mat, init_p_inds, init_n_inds, real_labels, n_iters_pu, n_neighbors=None,
):


    if n_neighbors is None:
        n_neighbors = knn_dist_mat.shape[1]

    knn_dists = knn_dist_mat[:, :n_neighbors]
    knn_inds = knn_inds_mat[:, :n_neighbors]
    n_examples = len(knn_dists)

    # initialization

    sttree = STTree(init_p_inds)
    labels_by_ind = -np.ones(n_examples, dtype=np.int8)
    labels_by_ind[init_p_inds] = 1
    labels_by_ind[init_n_inds] = 0

    valid_by_row = np.ones((n_examples, n_neighbors), dtype=np.int8)
    knn_dists[init_n_inds] = np.inf
    valid_by_row[init_n_inds] = 0

    non_init_n_inds = np.setdiff1d(np.arange(n_examples), init_n_inds, assume_unique=True)
    invalid = np.where(np.isin(knn_inds[non_init_n_inds], np.concatenate((init_p_inds, init_n_inds))))
    knn_dists[non_init_n_inds[invalid[0]], invalid[1]] = np.inf
    valid_by_row[non_init_n_inds[invalid[0]], invalid[1]] = 0

    u_knn_inds, all_i_in_knn_inds = unique_with_all_indices_1d(knn_inds.flatten())
    i1_in_knn_inds_map = dict(zip(u_knn_inds, all_i_in_knn_inds))
    del u_knn_inds, all_i_in_knn_inds

    p_knn_cursors = np.argmax(valid_by_row[init_p_inds], axis=1)    # the first valid indice in each row corresponding to a known positive example
    p_inds = deepcopy(init_p_inds)

    ranked_inds, nn_in_p, pu_dists = np.array([]).astype(int), np.array([]).astype(int), np.array([]).astype(int)

    # main loop
    for i in range(n_iters_pu):  # NOTE: this allows for the final example to be ranked (otherwise, we need to further minus 1)):

        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}/{n_iters_pu}")

        # tic = perf_counter()
        i_p = np.argmin(knn_dists[p_inds, p_knn_cursors])
        next_nn_in_p = p_inds[i_p]
        pu_dist = knn_dists[next_nn_in_p, p_knn_cursors[i_p]]
        if pu_dist == np.inf:
            print(f"WARNING: Early stopping at iteration {i + 1}/{n_iters_pu}")
            break
        next_p = knn_inds[next_nn_in_p, p_knn_cursors[i_p]]

        sttree.append(next_p, next_nn_in_p)

        p_inds = np.append(p_inds, next_p)
        ranked_inds = np.append(ranked_inds, next_p)
        nn_in_p = np.append(nn_in_p, next_nn_in_p)
        pu_dists = np.append(pu_dists, pu_dist)

        # move the knn cursors
        next_i1_in_knn_inds = i1_in_knn_inds_map[next_p]  # next_p must be in the keys of i_in_knn_inds_map
        next_rows_in_knn_inds, next_cols_in_knn_inds = \
            next_i1_in_knn_inds // n_neighbors, next_i1_in_knn_inds % n_neighbors
        knn_dists[next_rows_in_knn_inds, next_cols_in_knn_inds] = np.inf    # NOTE: this line can NOT be omitted!
        valid_by_row[next_rows_in_knn_inds, next_cols_in_knn_inds] = 0
        p_knn_cursors = np.argmax(valid_by_row[p_inds], axis=1)  # either the first valid indice of the row, or 0 in the case where all indices in the row is invalid

    real_labels_ranked_inds = real_labels[ranked_inds]
    real_labels_nn_in_p = real_labels[nn_in_p]

    return sttree, labels_by_ind, ranked_inds, nn_in_p, pu_dists, real_labels_ranked_inds, real_labels_nn_in_p


def apply_sc(
        ranked_inds, init_p_inds, pu_dists, n_examples, sc, bhrk_card=None, gbtrm_beta=None, all_tss=None,
        active_budget=0, real_labels=None, ori_sttree=None, ori_labels_by_ind=None, ori_sc_queried_inds=None,
):

    '''
    Apply non-active SC
    '''

    tic_non_active = perf_counter()

    ranked_inds_ = np.concatenate((init_p_inds, ranked_inds))   # PUSC requires ranked_inds_ to include the initially labeled inds
    init_n_p = len(init_p_inds)

    pusc = PUSC(len(ranked_inds_), init_n_p)
    if sc == "RW":
        pre_num_p = pusc.sc_rw(pu_dists)
    elif sc == "BHRK":
        pre_num_p = pusc.sc_bhrk(all_tss[ranked_inds_], np.arange(len(ranked_inds_)), bhrk_card)
    elif sc == "GBTRM":
        pre_num_p = np.array(pusc.sc_gbtrm(pu_dists, gbtrm_beta))  # NOTE: there are 5 of them

        if pre_num_p[0] < 0:
            non_active_time, inter_active_query_times = perf_counter() - tic_non_active, np.array([])
            predicted_labels = [None, None, None, None, None]
            sc_queried_inds = np.array([]).astype(int)
            estimated_prf = [(-1, -1, -1)] * 5
            return predicted_labels, sc_queried_inds, estimated_prf, non_active_time, inter_active_query_times
        predicted_labels = np.zeros((5, n_examples), dtype=np.int8)
        for i_gbtrm in range(5):
            cur_est_num_p_in_u = pre_num_p[i_gbtrm] - init_n_p
            cur_predicted_p_inds = ranked_inds[:cur_est_num_p_in_u]
            predicted_labels[i_gbtrm, cur_predicted_p_inds] = 1
            predicted_labels[i_gbtrm, init_p_inds] = 1
    else:
        raise ValueError(f"Stopping criterion {sc} not yet supported!")

    # These only apply to cases where sc is not GBTRM.
    if sc != 'GBTRM':
        est_num_p_in_u = pre_num_p - init_n_p
        predicted_p_inds = ranked_inds[:est_num_p_in_u]
        predicted_labels = np.zeros(n_examples, dtype=np.int8)
        predicted_labels[predicted_p_inds] = 1
        predicted_labels[init_p_inds] = 1

    non_active_time = perf_counter() - tic_non_active

    '''
    Active
    '''

    tic_active_overhead = perf_counter()
    sttree = deepcopy(ori_sttree)
    labels_by_ind = deepcopy(ori_labels_by_ind)
    sc_queried_inds = deepcopy(ori_sc_queried_inds)

    inter_active_query_times = np.array([])

    do_active = (active_budget > 0)
    if do_active:
        chains = sttree.get_chains(labels_by_ind)

        inf_by_ind = sttree.get_inf_by_ind(chains, n_examples)
        if sc_queried_inds is None:
            sc_queried_inds = np.array([]).astype(int)

        non_active_time += perf_counter() - tic_active_overhead

        tic_active = None
        while len(chains) > 0 and active_budget > 0:

            # get the chain to query
            tgt_chain_val, tgt_chain_score = -1, -1
            for key, chain_val in chains.items():

                chain_score = np.sum(inf_by_ind[chain_val]) / (np.log(len(chain_val)) + np.finfo(float).eps)
                assert inf_by_ind[chain_val][-1] == 0
                if np.log(len(chain_val)) == 0:
                    assert chain_score == 0
                else:
                    assert chain_score > 0

                if chain_score > tgt_chain_score:
                    tgt_chain_val, tgt_chain_score = deepcopy(chain_val), chain_score
            tgt_leaf_ind = tgt_chain_val[-1]

            # query the leaf label
            q_label, active_budget, iaq_time, tic_active = \
                one_query(tgt_leaf_ind, active_budget, real_labels, timing=True, prev_tic_active=tic_active)
            if iaq_time is not None:
                inter_active_query_times = np.append(inter_active_query_times, iaq_time)
            sc_queried_inds = np.append(sc_queried_inds, tgt_leaf_ind)

            # do rollback as necessary
            if q_label == 1:  # The leaf label is 1, no need to do rollback
                sttree, chains, inf_by_ind, labels_by_ind = \
                    update_variables_active_sc(tgt_chain_val, sttree, chains, inf_by_ind, labels_by_ind)
                continue

            # rollback code
            start, finish = 0, len(tgt_chain_val) - 1
            queried_in_chain = {tgt_chain_val[-1]: 0}
            i_bsf = 0
            while start <= finish:
                i_query = (start + finish) // 2
                ind = tgt_chain_val[i_query]
                assert labels_by_ind[ind] == -1
                label = -1 if ind not in queried_in_chain.keys() else queried_in_chain[ind]

                if label == -1:
                    if active_budget == 0:
                        cur_queried_inds = list(queried_in_chain.keys())
                        cur_queried_inds = [ind for ind in cur_queried_inds if ind != tgt_chain_val[-1]]
                        break
                    label, active_budget, iaq_time, tic_active = \
                        one_query(ind, active_budget, real_labels, timing=True, prev_tic_active=tic_active)
                    inter_active_query_times = np.append(inter_active_query_times, iaq_time)
                    queried_in_chain[ind] = label
                assert label != 2
                if label == 1:
                    if i_bsf < i_query + 1:
                        i_bsf = i_query + 1
                    start = i_query + 1
                    continue

                # label == 0
                parent_ind = sttree.get_parent_ind(ind)
                parent_label = labels_by_ind[parent_ind] if parent_ind not in queried_in_chain.keys() \
                    else queried_in_chain[parent_ind]
                if parent_label == -1:
                    if active_budget == 0:
                        cur_queried_inds = list(queried_in_chain.keys())
                        cur_queried_inds = [ind for ind in cur_queried_inds if ind != tgt_chain_val[-1]]
                        break
                    parent_label, active_budget, iaq_time, tic_active = \
                        one_query(parent_ind, active_budget, real_labels, timing=True, prev_tic_active=tic_active)
                    inter_active_query_times = np.append(inter_active_query_times, iaq_time)
                    queried_in_chain[parent_ind] = parent_label

                if parent_label in (1, 2):
                    cur_queried_inds = list(queried_in_chain.keys())
                    cur_queried_inds = [ind for ind in cur_queried_inds if ind != tgt_chain_val[-1]]
                    break

                finish = i_query - 1

            if len(cur_queried_inds) > 0:
                sc_queried_inds = np.append(sc_queried_inds, cur_queried_inds)

            sttree, chains, inf_by_ind, labels_by_ind = \
                update_variables_active_sc(
                    tgt_chain_val, sttree, chains, inf_by_ind, labels_by_ind, i_rollback_in_chain=i_bsf)

    one_inds = np.where(labels_by_ind == 1)[0]
    zero_inds = np.where(labels_by_ind == 0)[0]

    # estimate the performance of the current sc, using active queried and inferred results as pseudo-groundtruth
    if do_active:
        active_pred_all = np.concatenate(
            (np.ones(len(one_inds)), np.zeros(len(zero_inds)))
        )

        if sc != 'GBTRM':
            non_active_pred_all = np.concatenate(
                (predicted_labels[one_inds], predicted_labels[zero_inds])
            )
            estimated_prf = prf(active_pred_all, non_active_pred_all)
        else:
            estimated_prf = []
            for cur_predicted_labels in predicted_labels:
                non_active_pred_all = np.concatenate(
                    (cur_predicted_labels[one_inds], cur_predicted_labels[zero_inds])
                )
                estimated_prf.append(prf(active_pred_all, non_active_pred_all))

        one_inds_infer = np.setdiff1d(one_inds, np.concatenate((init_p_inds, sc_queried_inds)))
        zero_inds_infer = np.setdiff1d(zero_inds, np.concatenate((init_n_inds, sc_queried_inds)))

    else:
        if sc != 'GBTRM':
            estimated_prf = (-1, -1, -1)
        else:
            estimated_prf = [(-1, -1, -1)] * 5

        one_inds_infer = one_inds
        zero_inds_infer = zero_inds

    if sc != 'GBTRM':
        predicted_labels[one_inds] = 1
        predicted_labels[zero_inds] = 0
    else:
        predicted_labels[:, one_inds] = 1
        predicted_labels[:, zero_inds] = 0

    return predicted_labels, sc_queried_inds, np.array(estimated_prf), non_active_time, inter_active_query_times, \
           one_inds_infer, zero_inds_infer


def get_prf(
        real_labels: np.ndarray, predicted_labels: Union[np.ndarray, None]):

    if predicted_labels is None:
        return np.array([-1, -1, -1])

    return np.array(list(prf(real_labels, predicted_labels)))


if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument("data_id", choices=('MO1', 'MO2', 'MO3'), help='Data ID')
    parser.add_argument("total_query_proportion", type=float,
                        help="Proportion of data used for active queries, "
                             "including initiation and enhancement/stopping-criterion")
    parser.add_argument("active_init_proportion", type=float,
                        help="Proportion of queries used for active initialization")
    # parser.add_argument("sc_family", choices=("vanilla", "vanilla_active"),
    #                     help="The family of stopping criteria to use. "
    #                          "'vanilla' means only using the vanilla non-active SC. "
    #                          "'vanilla_active' means using ASCENSION-enhanced SC")
    parser.add_argument("rand_seed", type=int, help="Random seed for the random query strategy for active initialization. Not effective in other cases.")
    parser.add_argument("init_query_strategies", nargs="+",
                        help="Combination of initial active query strategies")

    parser.add_argument('--data_path', default=f'{main_dir}/data', help='Path to input dataset.')
    parser.add_argument('--dist_mat_path', default=f'{main_dir}/distances',
                        help='Path to pre-computed distance matrices')
    parser.add_argument('--result_path', default=f'{main_dir}/result', help='Result output path.')
    parser.add_argument("--max_positive_proportion", type=float, default=0.5,
                        help="The maximum possible proportion of positive examples in the entire dataset,"
                             "max_interval_iters = n_examples * max_positive_proportion - init_n_p")

    ############################

    args = parser.parse_args()

    data_id = args.data_id
    total_query_proportion = args.total_query_proportion
    active_init_proportion = args.active_init_proportion
    rand_seed = args.rand_seed
    init_query_strategies = args.init_query_strategies
    # sc_family = args.sc_family

    data_path = args.data_path
    dist_mat_path = args.dist_mat_path
    save_path = args.result_path
    max_positive_proportion = args.max_positive_proportion
    del args

    samp_rate = 512.

    # if 'active' not in sc_family and active_init_proportion != 1:
    #     raise ValueError('active_init_proportion must be set to 1.0 if active enhancement is not used.')
    '''
    Setup
    '''


    # load the data
    print('Loading data...', end='')
    tic = perf_counter()
    data_fname = f"{data_id}.pkl"
    full_data_fname = os.path.join(data_path, data_fname)
    train_examples, val_examples, test_examples, train_bi_labels, val_bi_labels, test_bi_labels = \
        FileReader.load_pickle(full_data_fname)
    n_train, n_val = len(train_examples), len(val_examples)

    real_labels = np.concatenate((train_bi_labels, val_bi_labels))
    n_examples = len(real_labels)
    init_max_n_queries = n_examples
    real_n_p = len(np.where(real_labels == 1)[0])
    init_max_n_queries = max(init_max_n_queries, real_n_p)
    print('Done.')

    # get amplitudes
    print('Calculating amplitudes...', end='')
    amplitudes = np.concatenate(
        (
            np.max(train_examples, axis=1) - np.min(train_examples, axis=1),
            np.max(val_examples, axis=1) - np.min(val_examples, axis=1),
        )
    )
    print('Done.')

    # load the knn matrices
    print("Loading KNN distances...", end="")
    knn_dist_fname = os.path.join(dist_mat_path, f"knn_dists_{data_id}.pkl")
    knn_dist_mat = FileReader.load_pickle(knn_dist_fname)
    knn_inds_fname = os.path.join(dist_mat_path, f"knn_inds_{data_id}.pkl")
    knn_inds_mat = FileReader.load_pickle(knn_inds_fname)
    print('Done.')

    init_q_strategies = []
    for i, strategy in enumerate(init_query_strategies):
        if strategy == "None":
            break
        if strategy == "random":
            strategy = f"random_{rand_seed}"
        init_q_strategies.append(strategy)
    assert len(init_q_strategies) == 1

    # load active initialization results
    print('Loading active initialization results...', end='')
    fname = os.path.join(save_path, f"Module_1_active_init_results_{data_id}_{init_q_strategies[0]}.pkl")
    init_queried_inds, init_queried_labels, _, _ = FileReader.load_pickle(fname)
    print('Done.')

    total_n_queries = np.rint(total_query_proportion * n_examples).astype(int)
    init_n_queries_pu = np.rint(active_init_proportion * total_query_proportion * n_examples).astype(int)
    init_n_queries_pu = max(init_n_queries_pu, np.where(init_queried_labels == 1)[0][0] + 1)
    enhance_n_queries_pu = total_n_queries - init_n_queries_pu
    print(f"\nAmong all {n_examples} examples, a total of {total_n_queries} will be queried, "
          f"among which {init_n_queries_pu} are for initialization while {enhance_n_queries_pu} are for enhancement.\n")

    init_queried_inds = init_queried_inds[:init_n_queries_pu]
    init_queried_labels = init_queried_labels[:init_n_queries_pu]
    init_p_inds = init_queried_inds[np.where(init_queried_labels == 1)]
    init_n_inds = init_queried_inds[np.where(init_queried_labels == 0)]

    assert len(init_p_inds) > 0

    init_n_p, init_n_n = len(init_p_inds), len(init_n_inds)
    max_n_iters_pu = np.rint(n_examples * max_positive_proportion).astype(int) - init_n_p
    assert max_n_iters_pu > len(np.where(real_labels == 1)[0]) - init_n_p

    print('Ranking U examples with ST-1NN...')
    ori_sttree, ori_labels_by_ind, ranked_inds, nn_in_p, pu_dists, real_labels_ranked_inds, \
    real_labels_nn_in_p, = \
        st1nn_no_sc(
            knn_dist_mat, knn_inds_mat, init_p_inds, init_n_inds, real_labels, max_n_iters_pu,
        )
    print('Done!')


    print(f'Applying GBTRM stopping criteria (SC) with active enhancement...')
    all_estimated_prf, all_predicted_labels = [], []
    for gbtrm_beta in (.1, .2, .3, .4, .5):
        scs = [f"GBTRM_{i}_{gbtrm_beta}" for i in range(1, 6, 1)]
        print(f"Current GBTRM Beta: {gbtrm_beta}")
        predicted_labels, sc_queried_inds, estimated_prf, non_active_time, inter_active_query_times, \
        one_inds_infer, zero_inds_infer = \
            apply_sc(
            ranked_inds, init_p_inds, pu_dists, n_examples, "GBTRM", gbtrm_beta=gbtrm_beta,
            ori_sttree=ori_sttree, ori_labels_by_ind=ori_labels_by_ind, active_budget=enhance_n_queries_pu,
            real_labels=real_labels, ori_sc_queried_inds=None,
        )

        all_estimated_prf.append(estimated_prf)
        all_predicted_labels += list(predicted_labels)
    all_estimated_prf = np.concatenate(tuple(all_estimated_prf))
    assert all_estimated_prf.shape == (25, 3)
    assert len(all_predicted_labels) == 25
    for predicted_labels in all_predicted_labels:
        assert predicted_labels is None or len(predicted_labels) == 3
    print('Done!')

    print('Selecting the best stopping criterion...', end='')
    all_estimated_f = all_estimated_prf[:, -1]
    selected_ind, selected_est_f = np.argmax(all_estimated_f), np.max(all_estimated_f)

    oracle_best_f = -1
    for predicted_labels in all_predicted_labels:
        oracle_best_f = max(oracle_best_f, get_prf(real_labels, predicted_labels)[-1])

    mean_time, std_time = np.mean(inter_active_query_times), np.std(inter_active_query_times)
    print('Done!\n')
    print(f'Transduction of the selected SC is {selected_est_f}, while that of the actual best SC is {oracle_best_f}.')
    print(f'User interaction response time is {mean_time} +- {std_time} seconds.\n')

    print('Filtering likely mislabeled examples...', end='')
    median_amp = np.median(amplitudes)
    good_inferred_p_inds = np.array([ind for ind in one_inds_infer if amplitudes[ind] > median_amp])
    good_inferred_n_inds = np.array([ind for ind in zero_inds_infer if amplitudes[ind] < median_amp])
    inds_to_keep = np.concatenate((
        init_p_inds, init_n_inds, sc_queried_inds, good_inferred_p_inds, good_inferred_n_inds
    ))
    train_inds_to_keep = inds_to_keep[inds_to_keep < n_train]
    val_inds_to_keep = inds_to_keep[inds_to_keep >= n_train] - n_train
    train_examples_to_keep, train_bi_labels_to_keep = \
        train_examples[train_inds_to_keep], train_bi_labels[train_inds_to_keep]
    val_examples_to_keep, val_bi_labels_to_keep = \
        val_examples[val_inds_to_keep], val_bi_labels[val_inds_to_keep]
    obj = train_examples_to_keep, train_bi_labels_to_keep, val_examples_to_keep, val_bi_labels_to_keep
    FileWriter.dump_pickle(obj, os.path.join(save_path, f'filtered_{data_id.pkl}'))
    print('Done!')

    print('All done!')


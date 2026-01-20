import numpy as np
from copy import deepcopy
from time import perf_counter
from util.evaluation_util import prf
from Z2H.ST1NN import st1nn_no_sc_gpu
from self_training_tree import SelfTrainingTree as STTree
from pu_stopping_criteria import PUStoppingCriteria as PUSC


# https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
def unique_with_all_indices_1d(array_1d):

    if array_1d.ndim != 1:
        raise Exception("unique_with_all_indices_1d only works for 1D ndarrays!")

    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(array_1d)

    # sorts records array so all unique elements are together
    sorted_array_1d = array_1d[idx_sort]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    u_vals, idx_start, count = np.unique(sorted_array_1d, return_counts=True, return_index=True)

    # splits the indices into separate arrays
    indices = np.split(idx_sort, idx_start[1:])

    return u_vals, indices


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

    tic_active = perf_counter()  # 标记本次query完成
    return q_label, budget, inter_active_query_time, tic_active


def update_variables(tgt_chain_val,
                     sttree: STTree, chains, inf_by_ind, p_inds, knn_dists,
                     valid_by_row, p_knn_cursors, ranked_inds, nn_in_p, pu_dists, labels_by_ind,
                     n_danger, i_rollback_in_chain=None, min_n_p_inds=10):

    if i_rollback_in_chain is None:
        one_inds = tgt_chain_val
        zero_inds = []
    else:
        one_inds = tgt_chain_val[:i_rollback_in_chain]
        rollback_ind = tgt_chain_val[i_rollback_in_chain]

        if n_danger > 0:
            ind = sttree.get_parent_ind(rollback_ind)
            two_inds = []
            for i_danger in range(n_danger):
                if ind == sttree.root_ind:
                    break
                if labels_by_ind[ind] != 2:
                    two_inds.append(ind)
                ind = sttree.get_parent_ind(ind)
            two_inds = np.array(two_inds)

        zero_inds = sttree.delete(ind=rollback_ind)

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

    if len(zero_inds) > 0:
        p_inds = p_inds[~np.isin(p_inds, zero_inds)]
        knn_dists[zero_inds, :] = np.inf
        valid_by_row[zero_inds, :] = 0
        p_knn_cursors = np.argmax(valid_by_row[p_inds], axis=1)

        mask = ~np.isin(ranked_inds, zero_inds)
        ranked_inds = ranked_inds[mask]
        nn_in_p = nn_in_p[mask]
        pu_dists = pu_dists[mask]

    if len(zero_inds) > 0:
        labels_by_ind[zero_inds] = 0
    if len(one_inds) > 0:
        labels_by_ind[one_inds] = 1


    if i_rollback_in_chain is not None and n_danger > 0:

        if len(two_inds) > 0:

            # further trim two_inds
            n_valid_p_inds = len(np.where(~np.isin(p_inds, two_inds))[0])
            if n_valid_p_inds < min_n_p_inds:
                i_start = min_n_p_inds - n_valid_p_inds
                if i_start <= len(two_inds) - 1:
                    two_inds = two_inds[i_start:]
                else:
                    two_inds = []

            if len(two_inds) > 0:
                p_inds = p_inds[~np.isin(p_inds, two_inds)]
                knn_dists[two_inds, :] = np.inf
                valid_by_row[two_inds, :] = 0
                p_knn_cursors = np.argmax(valid_by_row[p_inds], axis=1)
                labels_by_ind[two_inds] = 2

    return sttree, chains, inf_by_ind, p_inds, knn_dists, valid_by_row, p_knn_cursors, ranked_inds, \
           nn_in_p, pu_dists, labels_by_ind

def update_variables_active_sc(tgt_chain_val,
                               sttree: STTree, chains, inf_by_ind, labels_by_ind,
                               i_rollback_in_chain=None,):

    # get one_inds, two_inds and zero_inds, and update sttree
    if i_rollback_in_chain is None:
        one_inds = tgt_chain_val
        zero_inds = []
    else:
        one_inds = tgt_chain_val[:i_rollback_in_chain]
        rollback_ind = tgt_chain_val[i_rollback_in_chain]
        zero_inds = sttree.delete(ind=rollback_ind)

    # update chains and inf_by_ind，
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

def find_rollback(chain_val, budget, sttree:STTree, labels_by_ind, real_labels, budget_cushion=0):

    start, finish = 0, len(chain_val) - 1
    queried_in_chain = {chain_val[-1]: 0}
    i_bsf = 0
    while start <= finish:
        i_query = (start + finish) // 2
        ind = chain_val[i_query]
        assert labels_by_ind[ind] == -1
        label = -1 if ind not in queried_in_chain.keys() else queried_in_chain[ind]

        if label == -1:
            if budget == budget_cushion:
                cur_queried_inds = list(queried_in_chain.keys())
                cur_queried_inds = [ind for ind in cur_queried_inds if ind != chain_val[-1]]
                return i_bsf, budget, cur_queried_inds
            label, budget = one_query(ind, budget, real_labels)
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
            if budget == budget_cushion:
                cur_queried_inds = list(queried_in_chain.keys())
                cur_queried_inds = [ind for ind in cur_queried_inds if ind != chain_val[-1]]
                return i_bsf, budget, cur_queried_inds
            parent_label, budget = one_query(parent_ind, budget, real_labels)
            queried_in_chain[parent_ind] = parent_label

        if parent_label in (1, 2):
            i_bsf = i_query
            cur_queried_inds = list(queried_in_chain.keys())
            cur_queried_inds = [ind for ind in cur_queried_inds if ind != chain_val[-1]]
            return i_bsf, budget, cur_queried_inds

        finish = i_query - 1


def st1nn_active_one_class_no_sc(
        knn_dist_mat, knn_inds_mat, init_p_inds, init_n_inds, real_labels, n_iters_pu, n_neighbors=None,
        budget=1000,
        budget_cushion=0,
        max_interval_iters=-1,
        n_danger=10,
        min_n_p_inds=10,
):

    if max_interval_iters <= 0:
        max_interval_iters = int(1e10)  # Don't do active queries

    if n_neighbors is None:
        n_neighbors = knn_dist_mat.shape[1]
    else:
        n_neighbors = min(n_neighbors, knn_inds_mat.shape[1])

    knn_dists = deepcopy(knn_dist_mat[:, :n_neighbors])
    knn_inds = deepcopy(knn_inds_mat[:, :n_neighbors])
    n_examples = len(knn_dists)

    ### initialization

    sttree = STTree(init_p_inds)
    labels_by_ind = -np.ones(n_examples, dtype=np.int8)
    labels_by_ind[init_p_inds] = 1
    labels_by_ind[init_n_inds] = 0

    tic = perf_counter()

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

    ranked_inds, nn_in_p, pu_dists = np.array([]).astype(int), np.array([]).astype(int), np.array([]).astype(np.float32)
    queried_inds = np.array([]).astype(int)


    # main loop
    cnt_interval_iters = 0
    for i in range(n_iters_pu):

        if (i + 1) % 10000 == 0:
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
        knn_dists[next_rows_in_knn_inds, next_cols_in_knn_inds] = np.inf
        valid_by_row[next_rows_in_knn_inds, next_cols_in_knn_inds] = 0
        p_knn_cursors = np.argmax(valid_by_row[p_inds], axis=1)  # either the first valid indice of the row, or 0 in the case where all indices in the row is invalid

        cnt_interval_iters += 1
        if cnt_interval_iters != max_interval_iters:  # 尚未达到active querying条件
            continue

        if budget == 0:
            continue

        print(f"Conducting active querying in iteration {i + 1}/{n_iters_pu} with budget = {budget}.")

        # get initial chains and inf_by_ind
        chains = sttree.get_chains(labels_by_ind)

        inf_by_ind = sttree.get_inf_by_ind(chains, n_examples)

        while len(chains) > 0 and budget > 0:

            tgt_chain_val, tgt_chain_score = -1, -1
            for key, chain_val in chains.items():

                chain_score = np.sum(inf_by_ind[chain_val]) / (np.log(len(chain_val)) + np.finfo(float).eps)
                if chain_score > tgt_chain_score:
                    tgt_chain_val, tgt_chain_score = deepcopy(chain_val), chain_score
            tgt_leaf_ind = tgt_chain_val[-1]

            # query the leaf label
            q_label, budget = one_query(tgt_leaf_ind, budget, real_labels)
            queried_inds = np.append(queried_inds, tgt_leaf_ind)

            # do rollback as necessary
            if q_label == 1:    # The leaf label is 1, no need to do rollback
                sttree, chains, inf_by_ind, p_inds, knn_dists, valid_by_row, p_knn_cursors, ranked_inds, \
                nn_in_p, pu_dists, labels_by_ind = \
                    update_variables(tgt_chain_val,
                                     sttree, chains, inf_by_ind, p_inds, knn_dists,
                                     valid_by_row, p_knn_cursors, ranked_inds, nn_in_p, pu_dists, labels_by_ind,
                                     n_danger)
                continue

            # the leaf label is 0, do rollback:
            i_rollback_in_chain, budget, cur_queried_inds = find_rollback(
                tgt_chain_val, budget, sttree, labels_by_ind, real_labels, budget_cushion=budget_cushion)
            if len(cur_queried_inds) > 0:
                queried_inds = np.append(queried_inds, cur_queried_inds)

            sttree, chains, inf_by_ind, p_inds, knn_dists, valid_by_row, p_knn_cursors, ranked_inds, \
            nn_in_p, pu_dists, labels_by_ind = \
                update_variables(tgt_chain_val,
                                 sttree, chains, inf_by_ind, p_inds, knn_dists,
                                 valid_by_row, p_knn_cursors, ranked_inds, nn_in_p, pu_dists, labels_by_ind,
                                 n_danger, i_rollback_in_chain=i_rollback_in_chain, min_n_p_inds=min_n_p_inds)

        cnt_interval_iters = 0  # reset interval iterations counter
        print(f"Querying complete with remaining budget = {budget}. "
              f"Number of remaining \"-1\" examples is {len(np.where(labels_by_ind == -1)[0])}")

        assert (labels_by_ind[ranked_inds] != 0).all()
        if len(ranked_inds) + len(np.where(labels_by_ind == 0)[0]) == n_examples:
            break

    total_time = perf_counter() - tic
    print(f"st1nn_active_one_class_no_sc ran for {perf_counter() - tic} seconds")

    real_labels_ranked_inds = real_labels[ranked_inds]
    real_labels_nn_in_p = real_labels[nn_in_p]

    return sttree, labels_by_ind, ranked_inds, nn_in_p, pu_dists, real_labels_ranked_inds, real_labels_nn_in_p, queried_inds, total_time


def apply_vanilla_sc_st1nn_one_class(
        real_labels, ranked_inds, init_p_inds, init_n_inds, pu_dists, n_examples, active_budget,
        ori_sttree, ori_labels_by_ind, ori_sc_queried_inds,
        all_gbtrm_beta=None

):

    if all_gbtrm_beta is None:
        all_gbtrm_beta = (.1, .2, .3, .4, .5)

    '''
    Non-active
    '''

    tic_non_active = perf_counter()

    ranked_inds_ = np.concatenate((init_p_inds, ranked_inds))   # PUSC requires ranked_inds_ to include the initially labeled inds
    init_n_p = len(init_p_inds)

    pusc = PUSC(len(ranked_inds_), init_n_p)

    all_pre_num_p = np.concatenate(
        [pusc.sc_gbtrm(pu_dists, beta) for beta in all_gbtrm_beta]
    )
    assert len(all_pre_num_p) == 5 * len(all_gbtrm_beta)
    assert (all_pre_num_p >= 0).any()

    all_pre_num_p = all_pre_num_p[all_pre_num_p >= 0]   # only keep the successful ones
    n_gbtrm_settings = len(all_pre_num_p)


    all_predicted_labels = np.zeros((n_gbtrm_settings, n_examples)).astype(int)
    for i_gbtrm in range(n_gbtrm_settings):
        cur_est_num_p_in_u = all_pre_num_p[i_gbtrm] - init_n_p
        cur_predicted_p_inds = ranked_inds[:cur_est_num_p_in_u]
        all_predicted_labels[i_gbtrm, cur_predicted_p_inds] = 1
        all_predicted_labels[i_gbtrm, init_p_inds] = 1

    non_active_time = perf_counter() - tic_non_active

    '''
    Active
    '''

    inter_active_query_times = np.array([])

    tic_active_overhead = perf_counter()
    sttree = deepcopy(ori_sttree)
    labels_by_ind = deepcopy(ori_labels_by_ind)
    sc_queried_inds = deepcopy(ori_sc_queried_inds)
    if sc_queried_inds is None:
        sc_queried_inds = np.array([]).astype(int)

    chains = sttree.get_chains(labels_by_ind)
    inf_by_ind = sttree.get_inf_by_ind(chains, n_examples)

    non_active_time += perf_counter() - tic_active_overhead

    tic_active = None
    while len(chains) > 0 and active_budget > 0:

        # get the chain to query
        tgt_chain_val, tgt_chain_score = -1, -1
        for key, chain_val in chains.items():

            chain_score = np.sum(inf_by_ind[chain_val]) / (np.log(len(chain_val)) + np.finfo(float).eps)
            if chain_score > tgt_chain_score:
                tgt_chain_val, tgt_chain_score = deepcopy(chain_val), chain_score  # 注意这个deepcopy不能省
        tgt_leaf_ind = tgt_chain_val[-1]

        assert len(np.intersect1d(init_p_inds, init_n_inds)) == 0
        M1_queried_inds = np.concatenate((init_p_inds, init_n_inds))
        if tgt_leaf_ind in M1_queried_inds:
            print(f'Bad leaf ind: {tgt_leaf_ind}')

        # query the leaf label
        q_label, active_budget, iaq_time, tic_active = \
            one_query(tgt_leaf_ind, active_budget, real_labels, timing=True, prev_tic_active=tic_active)
        if iaq_time is not None:
            inter_active_query_times = np.append(inter_active_query_times, iaq_time)
        sc_queried_inds = np.append(sc_queried_inds, tgt_leaf_ind)

        for scq_ind in sc_queried_inds:
            if scq_ind in M1_queried_inds:
                print(f'Bad scq_ind at Checkpoint 1: {scq_ind}')

        # do rollback as necessary
        if q_label == 1:  # The leaf label is 1, no need to do rollback
            sttree, chains, inf_by_ind, labels_by_ind = \
                update_variables_active_sc(tgt_chain_val, sttree, chains, inf_by_ind, labels_by_ind)
            continue

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

                if ind in M1_queried_inds:
                    print(f'Bad ind: {ind}')

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
            parent_ind = sttree.get_parent_ind(ind)  # 注意这里不能直接chain[i_query - 1] (因为chain首节点)
            parent_label = labels_by_ind[parent_ind] if parent_ind not in queried_in_chain.keys() \
                else queried_in_chain[parent_ind]  # 注意labels_by_ind[parent_ind]不一定为-1！

            if parent_label == -1:
                if active_budget == 0:
                    cur_queried_inds = list(queried_in_chain.keys())
                    cur_queried_inds = [ind for ind in cur_queried_inds if ind != tgt_chain_val[-1]]
                    break

                if parent_ind in M1_queried_inds:
                    print(f'Bad parent ind: {parent_ind}')

                parent_label, active_budget, iaq_time, tic_active = \
                    one_query(parent_ind, active_budget, real_labels, timing=True, prev_tic_active=tic_active)
                inter_active_query_times = np.append(inter_active_query_times, iaq_time)
                queried_in_chain[parent_ind] = parent_label

            if parent_label in (1, 2):
                i_bsf = i_query
                cur_queried_inds = list(queried_in_chain.keys())
                cur_queried_inds = [ind for ind in cur_queried_inds if ind != tgt_chain_val[-1]]
                break

            finish = i_query - 1

        if len(cur_queried_inds) > 0:
            sc_queried_inds = np.append(sc_queried_inds, cur_queried_inds)

        for scq_ind in sc_queried_inds:
            if scq_ind in M1_queried_inds:
                print(f'Bad scq_ind at Checkpoint 2: {scq_ind}')

        sttree, chains, inf_by_ind, labels_by_ind = \
            update_variables_active_sc(
                tgt_chain_val, sttree, chains, inf_by_ind, labels_by_ind, i_rollback_in_chain=i_bsf)

    """
    Determine AIEs and NIEs (in KTEF)
    """

    assert 2 not in labels_by_ind
    one_inds = np.where(labels_by_ind == 1)[0]
    zero_inds = np.where(labels_by_ind == 0)[0]
    nie_inds = np.where(labels_by_ind == -1)[0]
    assert len(np.intersect1d(nie_inds, np.concatenate((init_p_inds, init_n_inds, sc_queried_inds)))) == 0

    one_inds_infer = np.setdiff1d(one_inds, np.concatenate((init_p_inds, sc_queried_inds)))
    zero_inds_infer = np.setdiff1d(zero_inds, np.concatenate((init_n_inds, sc_queried_inds)))
    aie_inds = np.sort(np.concatenate((one_inds_infer, zero_inds_infer)))

    '''
    SC selection
    '''

    # estimate the performance of the current sc, using active queried and inferred results as pseudo-groundtruth
    real_infer = np.concatenate((real_labels[one_inds_infer], real_labels[zero_inds_infer]))
    active_pred_infer = np.concatenate(
        (np.ones(len(one_inds_infer)), np.zeros(len(zero_inds_infer)))
    )
    active_pred_all = np.concatenate(
        (np.ones(len(one_inds)), np.zeros(len(zero_inds)))
    )


    actual_prf_infer, actual_prf_all = [], []
    estimated_prf_infer, estimated_prf_all = [], []
    for predicted_labels in all_predicted_labels:
        non_active_pred_infer = np.concatenate((predicted_labels[one_inds_infer], predicted_labels[zero_inds_infer]))
        non_active_pred_all = np.concatenate((predicted_labels[one_inds], predicted_labels[zero_inds]))

        actual_prf_infer.append(prf(real_infer, non_active_pred_infer))
        actual_prf_all.append(prf(real_labels,
                                  predicted_labels))
        estimated_prf_infer.append(prf(active_pred_infer, non_active_pred_infer))
        estimated_prf_all.append(prf(active_pred_all, non_active_pred_all))

    estimated_f_all = [prf_tuple[-1] for prf_tuple in estimated_prf_all]
    selected_best_sc = np.argmax(estimated_f_all)
    prf_selected_best = actual_prf_all[selected_best_sc]

    actual_f_all = [prf_tuple[-1] for prf_tuple in actual_prf_all]
    actual_best_sc = np.argmax(actual_f_all)
    prf_actual_best = actual_prf_all[actual_best_sc]

    predicted_labels = all_predicted_labels[selected_best_sc]
    assert len(predicted_labels) == len(labels_by_ind)
    predicted_labels[one_inds] = 1
    predicted_labels[zero_inds] = 0

    assert (real_labels[np.intersect1d(one_inds, sc_queried_inds)] == 1).all()
    assert (real_labels[np.intersect1d(zero_inds, sc_queried_inds)] == 0).all()
    assert (predicted_labels[sc_queried_inds] == real_labels[sc_queried_inds]).all()

    enhanced_prf = prf(real_labels, predicted_labels)

    for scq_ind in sc_queried_inds:
        if scq_ind in np.concatenate((init_p_inds, init_n_inds)):
            print(f'Bad scq_ind at Checkpoint 3: {scq_ind}')
    assert len(np.intersect1d(sc_queried_inds, np.concatenate((init_p_inds, init_n_inds)))) == 0


    return (
        predicted_labels,
        sc_queried_inds, aie_inds, nie_inds,
        enhanced_prf,
        actual_prf_infer,
        actual_prf_all,
        actual_best_sc, prf_actual_best,
        estimated_prf_infer,
        estimated_prf_all,
        selected_best_sc, prf_selected_best,
        non_active_time,
        inter_active_query_times
    )


def st1nn_with_ascension(
        knn_dist_mat, knn_inds_mat, init_p_inds, init_n_inds, real_labels,
        max_n_iters_pu, n_neighbors_pu, enhance_n_queries_pu,
        budget_cushion=0,
        max_interval_iters=-1,
        n_danger=10,
        min_n_p_inds=10,
):

    n_examples = len(knn_dist_mat)

    # do it on the CPU
    ori_sttree_cpu, ori_labels_by_ind_cpu, ranked_inds_cpu, nn_in_p_cpu, pu_dists_cpu, real_labels_ranked_inds_cpu, \
        real_labels_nn_in_p_cpu, ori_sc_queried_inds_cpu, st1nn_time_cpu = \
        st1nn_active_one_class_no_sc(
            knn_dist_mat, knn_inds_mat, init_p_inds, init_n_inds, real_labels, max_n_iters_pu,
            n_neighbors=n_neighbors_pu, budget=0, budget_cushion=budget_cushion,
            max_interval_iters=max_interval_iters, n_danger=n_danger,
            min_n_p_inds=min_n_p_inds,
        )

    # Do it on the GPU
    ranked_inds, nn_in_p, pu_dists, st1nn_time_gpu = st1nn_no_sc_gpu(knn_dist_mat, knn_inds_mat,
        init_p_inds, init_n_inds,
        max_iters=max_n_iters_pu, n_neighbors=n_neighbors_pu)

    # Ascension initialization based on GPU results
    tic = perf_counter()

    sttree_tic = perf_counter()
    ori_sttree = STTree(init_p_inds)
    for next_p, next_nn_in_p in zip(ranked_inds, nn_in_p):
        ori_sttree.append(next_p, next_nn_in_p)
    print(f'STTree construction time: {perf_counter() - sttree_tic} seconds.')

    ori_labels_by_ind = -np.ones(n_examples, dtype=np.int8)
    ori_labels_by_ind[init_p_inds] = 1
    ori_labels_by_ind[init_n_inds] = 0
    ori_sc_queried_inds = np.array([]).astype(int)

    ascension_init_time = perf_counter() - tic

    # ascension
    (predicted_labels, sc_queried_inds, aie_inds, nie_inds, enhanced_prf, actual_prf_infer, actual_prf_all, actual_best_sc, f_actual_best,
     estimated_prf_infer, estimated_prf_all, selected_best_sc, f_selected_best, non_active_time, inter_active_query_times) \
        = (
        apply_vanilla_sc_st1nn_one_class(
            real_labels, ranked_inds, init_p_inds, init_n_inds, pu_dists, n_examples, enhance_n_queries_pu,
            ori_sttree, ori_labels_by_ind, ori_sc_queried_inds,
            all_gbtrm_beta=None,
        ))
    assert len(np.intersect1d(sc_queried_inds, np.concatenate((init_p_inds, init_n_inds)))) == 0

    non_active_time += ascension_init_time

    return (predicted_labels, sc_queried_inds, aie_inds, nie_inds, enhanced_prf, actual_prf_infer, actual_prf_all, actual_best_sc, f_actual_best,
     estimated_prf_infer, estimated_prf_all, selected_best_sc, f_selected_best, st1nn_time_cpu, st1nn_time_gpu, non_active_time, inter_active_query_times)


import os
import argparse
import numpy as np
from util.pytorch_util import set_seed

from util.file_util import FileReader, FileWriter, DirProcessor
from util.evaluation_util import prf
from util.seeg_file_util import load_fully_supervised_origin_data, to_seanet_format
from Z2H.Z2H_M1 import M1_Z2H
from Z2H.Z2H_M2 import st1nn_with_ascension
from Z2H.Z2H_M3 import M3_No_Filter, M3_Only_Manual, M3_KTEF
from Z2H.calc_knn_dists import calc_knn_from_ndarray
from util.miscellaneous_util import pack_vars_to_dict
import shlex
from pathlib import Path


def Z2H_parse_args(extern_params=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('data_id', type=str)
    parser.add_argument('--channel', type=int, default=None)

    parser.add_argument('--rand_seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')

    parser.add_argument('--timing_runs', type=int, default=None)

    parser.add_argument('--main_data_path', type=str, default=None)
    parser.add_argument('--main_dl_data_path', type=str, default=None)
    parser.add_argument('--main_z2h_save_path', type=str, default=None)

    """
    AL budget allocation
    """
    parser.add_argument("--total_query_proportion", type=float, required=True,
                        help="Proportion of data used for active queries, "
                             "including initiation and enhancement/stopping-criterion")
    parser.add_argument("--active_init_proportion", type=float, required=True,
                        help="Proportion of queries used for active initialization")


    """
    M1 parameters
    """

    parser.add_argument('--M1_method', type=str, nargs='+', required=True,
                        choices=M1_Z2H.valid_base_sampling_stategies)

    # Z2H-only parameters
    parser.add_argument('--M1_amp_calculator', type=str, default=None)
    parser.add_argument('--M1_dist_calculator', type=str, default='1nn')
    parser.add_argument('--M1_filter_threshold', type=str, default='median')
    parser.add_argument('--M1_adaptive_filter', type=str, default='knn')
    parser.add_argument('--M1_af_knn_k', type=int, default=1)

    parser.add_argument('--M1_channel_ranker', type=str, default=None)
    parser.add_argument('--M1_ch_select_warmup', type=str, default=None)


    """
    M2 parameters
    """
    parser.add_argument('--M2_method', type=str, choices=['ASCENSION'], default='ASCENSION')

    # for ASCENSION
    parser.add_argument("--M2_n_neighbors_pu", type=int, default=1000)
    parser.add_argument("--M2_max_positive_proportion", type=float, default=0.5)

    """
    M3 parameters
    """
    parser.add_argument('--M3_method', type=str, nargs='+', default=None,
                        choices=['no_filter', 'only_manual'] + list(M3_KTEF.valid_base_filter_strategies))
    parser.add_argument('--M3_amp_calculator', type=str, default=None)
    parser.add_argument('--M3_dist_calculator', type=str, default=None)
    parser.add_argument('--M3_amp_filter_th', type=str, default='median')
    parser.add_argument('--M3_dist_filter_th', type=str, default='median')

    if extern_params is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shlex.split(extern_params))

    if args.all_M3_methods is None:
        args.all_M3_methods = ['amp',]

    assert not (args.only_M1 and args.only_M1_M2)

    if any('amp' in M1_method for M1_method in args.all_M1_methods):
        assert args.M1_amp_calculator is not None
        if (args.channel is not None) or (args.M1_channel_ranker is not None):    # single-channel
            assert args.M1_amp_calculator in M1_Z2H.valid_single_channel_amp_calculators
        else:
            assert args.M1_amp_calculator in M1_Z2H.valid_multi_channel_amp_calculators
    else:
        assert args.M1_amp_calculator is None

    if any('dist' in M1_method for M1_method in args.all_M1_methods):
        assert args.M1_dist_calculator is not None
        if args.channel is not None:
            assert args.M1_dist_calculator in M1_Z2H.valid_single_channel_dist_calculators
        else:
            assert args.M1_dist_calculator in M1_Z2H.valid_multi_channel_dist_calculators
    else:
        assert args.M1_dist_calculator == '1nn'

    if (
            not any('_' in M1_method for M1_method in args.all_M1_methods
                    if M1_method in M1_Z2H.valid_base_sampling_stategies)
    ):
        assert args.M1_filter_threshold == 'median'
        assert args.M1_adaptive_filter == 'knn'
        assert args.M1_af_knn_k == 1

    if args.M1_adaptive_filter != 'knn':
        assert args.M1_af_knn_k == 1

    if ((not any(M1_method in M1_Z2H.valid_base_sampling_stategies for M1_method in args.all_M1_methods)) or
            (args.channel is not None)):
        assert (args.M1_channel_ranker is None) and (args.M1_ch_select_warmup is None)

    if args.only_M1 or args.only_M1_M2:
        assert args.all_M3_methods == ['amp',]

    if args.M1_amp_calculator is not None:
        assert args.M3_amp_calculator is None
    if args.M1_dist_calculator is not None:
        assert args.M3_dist_calculator is None

    if not any('amp' in M3_method for M3_method in args.all_M3_methods):
        assert args.M3_amp_calculator is None
    else:
        assert not ((args.M1_amp_calculator is None) and (args.M3_amp_calculator is None))

    if not any('dist' in M3_method for M3_method in args.all_M3_methods):
        assert args.M3_dist_calculator is None
    else:
        assert not ((args.M1_dist_calculator is None) and (args.M3_dist_calculator is None))

    return args


def run_M1(args, multi_channel_examples, real_labels, M1_method, save_path, timing_run_id=-1):

    print(f"****** M1 with {M1_method}  ******")

    result_fname = os.path.join(save_path, 'M1_results_0.pkl')
    runtime_fname = os.path.join(save_path, f'M1_time_run_id={timing_run_id}.pkl')

    result_saving_needed = args.overwrite or (not os.path.exists(result_fname))
    runtime_saving_needed = timing_run_id >= 0 and (args.overwrite or (not os.path.exists(runtime_fname)))
    if not (result_saving_needed or runtime_saving_needed):
        result_dict = FileReader.load_pickle(result_fname)
    else:
        if M1_method in M1_Z2H.valid_base_sampling_stategies:
            m1 = M1_Z2H(
                args.data_id, args.channel,
                multi_channel_examples, real_labels, M1_method, args.M1_amp_calculator, args.M1_dist_calculator,
                filter_threshold=args.M1_filter_threshold, adaptive_filter=args.M1_adaptive_filter,
                channel_ranker=args.M1_channel_ranker, ch_select_warmup=args.M1_ch_select_warmup,

                af_knn_k=args.M1_af_knn_k
            )
        else:
            raise NotImplementedError()

        m1.active_queries()
        if hasattr(m1, 'best_channel'):
            best_channel = m1.best_channel
        else:
            best_channel = args.channel

        first_p = np.where(m1.label_by_queried_ind == 1)[0][0] + 1
        p_at_n = {n: -1 if n > len(m1.label_by_queried_ind) else np.sum(m1.label_by_queried_ind[:n]) / n
                  for n in range(10, 110, 10)}
        result_dict = pack_vars_to_dict(queried_inds=m1.queried_inds, label_by_queried_ind=m1.label_by_queried_ind,
                                        best_channel=best_channel,
                                        response_times=m1.response_times,
                                        first_p=first_p, p_at_n=p_at_n,)
        if result_saving_needed:
            FileWriter.dump_pickle(result_dict, result_fname)
        if runtime_saving_needed:
            FileWriter.dump_pickle(m1.response_times, runtime_fname)


    M1_queried_inds_full, M1_label_by_queried_ind, M1_best_channel, M1_response_times  = (
        result_dict['queried_inds'], result_dict['label_by_queried_ind'], result_dict['best_channel'],
        result_dict['response_times'],
    )

    M1_first_p, M1_p_at_n = result_dict['first_p'], result_dict['p_at_n']
    print(f'M1_first_p: {M1_first_p}')
    for n, precision in M1_p_at_n.items():
        print(f'p@{n}: {precision}')

    knn_dist_mat, knn_inds_mat = calc_knn_from_ndarray(
        multi_channel_examples, channels=args.channel, k=args.M2_n_neighbors_pu)
    n_at_proportion = min(
        np.rint(len(real_labels) * args.total_query_proportion * args.active_init_proportion).astype(int),
        len(M1_queried_inds_full)
    )

    result_fname = os.path.join(save_path, 'M1_results_1.pkl')    # results related to query proportion
    result_saving_needed = args.overwrite or (not os.path.exists(result_fname))
    if not result_saving_needed:
        result_dict = FileReader.load_pickle(result_fname)
    else:
        p_at_proportion = np.sum(M1_label_by_queried_ind[:n_at_proportion]) / n_at_proportion
        r_at_proportion = (np.sum(M1_label_by_queried_ind[:n_at_proportion]) / np.sum(real_labels))
        mean_response_time, std_response_time = (np.mean(M1_response_times[:n_at_proportion]),
                                                 np.std(M1_response_times[:n_at_proportion]))
        result_dict = pack_vars_to_dict(
            p_at_proportion=p_at_proportion, r_at_proportion=r_at_proportion,
            mean_response_time=mean_response_time, std_response_time=std_response_time
        )
        FileWriter.dump_pickle(result_dict, result_fname)

    M1_p_at_proportion, M1_r_at_proportion = result_dict['p_at_proportion'], result_dict['r_at_proportion']
    print(f'M1_p_at_proportion (which is {n_at_proportion}): {M1_p_at_proportion}')
    print(f'M1_r_at_proportion (which is {n_at_proportion}): {M1_r_at_proportion}')

    return M1_queried_inds_full, M1_label_by_queried_ind, M1_best_channel, knn_dist_mat, knn_inds_mat


def run_M2(
        args, M1_queried_inds_full, M1_label_by_queried_ind, real_labels, knn_dist_mat, knn_inds_mat, save_path,
        timing_run_id=-1
):

    print("****** M2 ******")

    result_fname = os.path.join(save_path, 'M2_results.pkl')
    runtime_fname = os.path.join(save_path, f'M2_time_run_id={timing_run_id}.pkl')

    result_saving_needed = args.overwrite or (not os.path.exists(result_fname))
    runtime_saving_needed = timing_run_id >= 0 and (args.overwrite or (not os.path.exists(runtime_fname)))

    if not (result_saving_needed or runtime_saving_needed):
        result_dict = FileReader.load_pickle(result_fname)
    else:
        n_examples = len(real_labels)
        total_n_queries = np.rint(args.total_query_proportion * n_examples).astype(int)
        init_n_queries_pu = np.rint(args.active_init_proportion * args.total_query_proportion * n_examples).astype(
            int)
        init_n_queries_pu = max(init_n_queries_pu, np.where(M1_label_by_queried_ind == 1)[0][0] + 1)
        enhance_n_queries_pu = total_n_queries - init_n_queries_pu
        print(f"Among all {n_examples} examples, a total of {total_n_queries} are queried, "
              f"among which {init_n_queries_pu} are for initialization while {enhance_n_queries_pu} are for enhancement.")

        M1_queried_inds = M1_queried_inds_full[:init_n_queries_pu]
        init_queried_labels = M1_label_by_queried_ind[:init_n_queries_pu]
        init_p_inds = M1_queried_inds[np.where(init_queried_labels == 1)]
        init_n_inds = M1_queried_inds[np.where(init_queried_labels == 0)]

        assert len(init_p_inds) > 0

        init_n_p, init_n_n = len(init_p_inds), len(init_n_inds)
        max_n_iters_pu = np.rint(n_examples * args.M2_max_positive_proportion).astype(int) - init_n_p
        assert max_n_iters_pu > len(np.where(real_labels == 1)[0]) - init_n_p

        if args.M2_method == 'ASCENSION':
            (predicted_labels, M2_queried_inds, aie_inds, nie_inds, enhanced_prf, actual_prf_infer, actual_prf_all, actual_best_sc, prf_actual_best,
            estimated_prf_infer, estimated_prf_all, selected_best_sc, prf_selected_best, st1nn_time_cpu, st1nn_time_gpu, non_active_time, inter_active_query_times) = (
                st1nn_with_ascension(knn_dist_mat, knn_inds_mat, init_p_inds, init_n_inds, real_labels,
                                     max_n_iters_pu, args.M2_n_neighbors_pu, enhance_n_queries_pu)
            )
        else:
            raise NotImplementedError
        result_dict = pack_vars_to_dict(
            predicted_labels=predicted_labels,
            M2_queried_inds=M2_queried_inds, aie_inds=aie_inds, nie_inds=nie_inds,
            enhanced_prf=enhanced_prf,
            actual_prf_infer=actual_prf_infer, actual_prf_all=actual_prf_all,
            actual_best_sc=actual_best_sc, prf_actual_best=prf_actual_best,
            estimated_prf_infer=estimated_prf_infer, estimated_prf_all=estimated_prf_all,
            selected_best_sc=selected_best_sc, prf_selected_best=prf_selected_best,
            st1nn_time_cpu=st1nn_time_cpu, st1nn_time_gpu=st1nn_time_gpu,
            non_active_time=non_active_time, inter_active_query_times=inter_active_query_times,
            M1_queried_inds=M1_queried_inds,
            init_n_queries_pu=init_n_queries_pu, enhance_n_queries_pu=enhance_n_queries_pu
        )
        if result_saving_needed:
            FileWriter.dump_pickle(result_dict, result_fname)
        if runtime_saving_needed:
            FileWriter.dump_pickle(
                {
                    'M1_to_M2_time_cpu': non_active_time + st1nn_time_cpu,
                    'M1_to_M2_time_gpu': non_active_time + st1nn_time_gpu,
                    'M2_user_response_times': inter_active_query_times,
                },
                runtime_fname
            )

    M2_predicted_labels, M1_queried_inds, M2_queried_inds, M2_aie_inds = (
        result_dict['predicted_labels'], result_dict['M1_queried_inds'], result_dict['M2_queried_inds'], result_dict['aie_inds'])
    assert len(np.intersect1d(M1_queried_inds, M2_queried_inds)) == 0

    M2_enhanced_prf, M2_prf_selected_best, M2_prf_actual_best = (result_dict['enhanced_prf'],
                                                                 result_dict['prf_selected_best'], result_dict['prf_actual_best'])

    return M1_queried_inds, M2_queried_inds, M2_predicted_labels, M2_aie_inds, M2_enhanced_prf, M2_prf_selected_best, M2_prf_actual_best


def run_M3(args, multi_channel_examples, channel, real_labels,
           M1_queried_inds, M2_queried_inds, M2_aie_inds, M2_predicted_labels,
           knn_dist_mat, knn_inds_mat, M3_method, save_path):

    print(f"****** M3: {M3_method}******")

    result_fname = os.path.join(save_path, 'M3_results.pkl')
    if (not args.overwrite) and (os.path.exists(result_fname)):
        result_dict = FileReader.load_pickle(result_fname)
    else:
        M3_amp_calculator = args.M1_amp_calculator if args.M1_amp_calculator is not None else args.M3_amp_calculator
        M3_dist_calculator = args.M1_dist_calculator if args.M1_dist_calculator is not None else args.M3_dist_calculator

        if M3_method == 'no_filter':
            m3 = M3_No_Filter(multi_channel_examples)
        elif M3_method == 'only_manual':
            m3 = M3_Only_Manual(multi_channel_examples, M1_queried_inds, M2_queried_inds)
        elif M3_method in M3_KTEF.valid_base_filter_strategies:
            m3 = M3_KTEF(
                multi_channel_examples, channel, M1_queried_inds, M2_queried_inds, M2_aie_inds,M2_predicted_labels,
                M3_method, amp_calculator=M3_amp_calculator, dist_calculator=M3_dist_calculator,
                amp_filter_th=args.M3_amp_filter_th, dist_filter_th=args.M3_dist_filter_th,
                knn_dist_mat=knn_dist_mat, knn_inds_mat=knn_inds_mat,
            )
        else:
            raise NotImplementedError()

        m3.select_inds()
        M3_selected_inds = m3.selected_inds

        M3_real_labels, M3_predicted_labels = (
            real_labels[M3_selected_inds], M2_predicted_labels[M3_selected_inds])

        result_dict = pack_vars_to_dict(
            selected_inds=m3.selected_inds,
            proportion_preserved=len(m3.selected_inds) / len(real_labels),

            prf_before_selection=prf(real_labels, M2_predicted_labels),
            actual_np_before_selection=len(np.where(real_labels == 1)[0]),
            actual_nn_before_selection=len(np.where(real_labels == 0)[0]),
            pred_np_before_selection=len(np.where(M2_predicted_labels == 1)[0]),
            pred_nn_before_selection=len(np.where(M2_predicted_labels == 0)[0]),

            prf_after_selection=prf(M3_real_labels, M3_predicted_labels),
            actual_np_after_selection=len(np.where(M3_real_labels == 1)[0]),
            actual_nn_after_selection=len(np.where(M3_real_labels == 0)[0]),
            pred_np_after_selection=len(np.where(M3_predicted_labels == 1)[0]),
            pred_nn_after_selection=len(np.where(M3_predicted_labels == 0)[0]),

        )
        FileWriter.dump_pickle(result_dict, result_fname)

    for key, val in result_dict.items():
        if key == 'selected_inds':
            continue

        print(f'M3_{key} = {val}')

    M3_selected_inds, M3_prf_before_selection, M3_prf_after_selection = (
        result_dict['selected_inds'], result_dict['prf_before_selection'], result_dict['prf_after_selection'])

    return M3_selected_inds, M3_prf_before_selection, M3_prf_after_selection


def get_selected_tv_data(
        examples, labels, selected_inds, proportion_train,
):
    selected_examples, selected_labels = examples[selected_inds], labels[selected_inds]
    p_idx_selected = np.where(selected_labels == 1)[0]
    n_idx_selected = np.where(selected_labels == 0)[0]
    np_selected, nn_selected = map(len, (p_idx_selected, n_idx_selected))
    assert np_selected >= 2 and nn_selected >= 2

    np_train_selected = max(np.rint(np_selected * proportion_train).astype(int),
                            1)  # at least one positive example for train
    np_train_selected = min(np_train_selected, np_selected - 1)  # at least one positive example for val
    nn_train_selected = max(np.rint(nn_selected * proportion_train).astype(int),
                            1)  # at least one negative example for train
    nn_train_selected = min(nn_train_selected, nn_selected - 1)  # at least one negative example for val
    np.random.shuffle(p_idx_selected)
    np.random.shuffle(n_idx_selected)
    train_p_idx, val_p_idx = p_idx_selected[:np_train_selected], p_idx_selected[np_train_selected:]
    train_n_idx, val_n_idx = n_idx_selected[:nn_train_selected], n_idx_selected[nn_train_selected:]

    train_p_ex, val_p_ex = selected_examples[train_p_idx], selected_examples[val_p_idx]
    train_n_ex, val_n_ex = selected_examples[train_n_idx], selected_examples[val_n_idx]
    train_p_labels, val_p_labels = selected_labels[train_p_idx], selected_labels[val_p_idx]
    assert (train_p_labels == np.ones_like(train_p_labels)).all()
    assert (val_p_labels == np.ones_like(val_p_labels)).all()
    train_n_labels, val_n_labels = selected_labels[train_n_idx], selected_labels[val_n_idx]
    assert (train_n_labels == np.zeros_like(train_n_labels)).all()
    assert (val_n_labels == np.zeros_like(val_n_labels)).all()

    return train_p_ex, train_n_ex, val_p_ex, val_n_ex


def save_to_seanet_format(
    tv_examples, test_examples, tv_labels, test_labels,
    selected_inds, proportion_train,
    main_dl_data_path,
    data_id, channel_id,
    overwrite=False
):
    train_p_ex, train_n_ex, val_p_ex, val_n_ex = (
        get_selected_tv_data(tv_examples, tv_labels, selected_inds, proportion_train))

    test_p_ex = test_examples[test_labels == 1]
    test_n_ex = test_examples[test_labels == 0]

    to_seanet_format('train', f'{data_id}_{channel_id}', train_p_ex, train_n_ex,
                     main_save_path=main_dl_data_path, overwrite=overwrite)
    to_seanet_format('valid', f'{data_id}_{channel_id}', val_p_ex, val_n_ex, main_save_path=main_dl_data_path,
                     overwrite=overwrite)
    to_seanet_format('test', f'{data_id}_{channel_id}', test_p_ex, test_n_ex, main_save_path=main_dl_data_path,
                     overwrite=overwrite)


if __name__ == '__main__':

    args = Z2H_parse_args()
    set_seed(args.rand_seed)

    main_z2h_save_path = args.main_z2h_save_path
    if main_z2h_save_path is None:
        main_z2h_save_path = str(Path(__file__).resolve().parent.parent.parent / "Z2H")

    main_data_path = args.main_data_path
    if main_data_path is None:
        main_data_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "data")

    main_dl_data_path = args.main_dl_data_path
    if main_dl_data_path is None:
        main_dl_data_path = os.path.join(main_data_path, 'Z2H_dl_data')

    channel_id = args.channel if args.channel is not None else 'all_ch'
    z2h_save_path = os.path.join(main_z2h_save_path, f'{args.data_id}_{channel_id}')
    DirProcessor.create_dir(z2h_save_path)

    """
    Load data
    """

    train_examples, val_examples, test_examples, train_bi_labels, val_bi_labels, test_bi_labels = (
        load_fully_supervised_origin_data(args.data_id, include_test=True, main_data_path=main_data_path))    # 注意：不要设置channel（因为这里必须是multi-channel）
    proportion_train = len(train_examples) / (len(train_examples) + len(val_examples))
    tv_examples = np.concatenate((train_examples, val_examples))
    real_labels = np.concatenate((train_bi_labels, val_bi_labels))

    if args.timing_runs is None:
        timing_run_ids = [-1]
    else:
        assert args.timing_runs > 0
        timing_run_ids = list(range(args.timing_runs))

    for timing_run_id in timing_run_ids:

        if timing_run_id > 0:
            print(f'\n************ timing_run_id = {timing_run_id} ************\n')

        """
        M1
        """

        M1_queried_inds_full, M1_label_by_queried_ind, M1_best_channel, knn_dist_mat, knn_inds_mat = (
            run_M1(args, tv_examples, real_labels, args.M1_method, z2h_save_path, timing_run_id=timing_run_id))

        """
        M2
        """
        M1_queried_inds, M2_queried_inds, M2_predicted_labels, M2_aie_inds, M2_enhanced_prf, M2_prf_selected_best, M2_prf_actual_best = (
            run_M2(args, M1_queried_inds_full, M1_label_by_queried_ind, real_labels, knn_dist_mat, knn_inds_mat, z2h_save_path,
                   timing_run_id=timing_run_id))

        assert (M2_predicted_labels[M1_queried_inds] == real_labels[M1_queried_inds]).all()
        assert (M2_predicted_labels[M2_queried_inds] == real_labels[M2_queried_inds]).all()

        """
        M3
        """

        M3_selected_inds, M3_prf_before_selection, M3_prf_after_selection = run_M3(
            args, tv_examples, M1_best_channel, real_labels, M1_queried_inds, M2_queried_inds, M2_aie_inds,
            M2_predicted_labels, knn_dist_mat, knn_inds_mat, args.M3_method, z2h_save_path)

    """
    Save for deep learning
    """

    tv_examples = np.concatenate((train_examples, val_examples))
    tv_examples = tv_examples if args.channel is None else tv_examples[:, args.channel, :]

    save_to_seanet_format(
        tv_examples, M2_predicted_labels, test_examples, test_bi_labels, M3_selected_inds, proportion_train,
        main_dl_data_path,
        args.data_id, channel_id,
        overwrite=args.overwrite
    )




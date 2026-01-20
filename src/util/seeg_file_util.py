from util.file_util import DirProcessor, FileReader
import os
import pickle
import subprocess
import numpy as np
from typing import Union
from pathlib import Path

n_channels_selector = {
    'MO1': 5,
    'MO2': 5,
    'MO3': 4,
}

ts_len_selector = {
    'MO1': 46,
    'MO2': 46,
    'MO3': 46,
}


def determine_n_channels(data_id):
    for data_id, n_channels in n_channels_selector.items():
        if data_id in data_id:
            return n_channels
    raise ValueError(f'Unknown number of channels for {data_id}. '
                     f'You need to update this info in n_channels_selector in seeg_file_util.py.')


def load_fully_supervised_origin_data(
        data_id, channels: Union[int, list, np.ndarray] = None, include_test=True, main_data_path=None):      # 注意：这里读进来的结果一定不能有任何随机性！

    if main_data_path is None:
        main_data_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "data")
    data_path = os.path.join(main_data_path, data_id)

    all_examples, all_bi_labels = [], []
    for tvt_id in ('train', 'valid', 'test'):   # 注意：这里是valid不是val

        if not include_test and tvt_id == 'test':
            continue

        with open(os.path.join(data_path, f'{tvt_id}.pkl'), 'rb') as f:
            ps, pl, ns, nl = pickle.load(f)

        with open(os.path.join(data_path, f'{tvt_id}_pn_inds.pkl'), 'rb') as f:
            p_inds, n_inds = pickle.load(f)

        # reconstruct train or valid examples
        order = np.argsort(np.concatenate((p_inds, n_inds)))
        assert (np.concatenate((p_inds, n_inds))[order] == np.arange(len(p_inds) + len(n_inds))).all()
        examples, bi_labels = np.concatenate((ps, ns)), np.concatenate((pl, nl))
        examples, bi_labels = examples[order], bi_labels[order]
        if channels is not None:
            examples = examples[:, channels, :]

        all_examples.append(examples.astype(np.float32))
        all_bi_labels.append(bi_labels)

    if not include_test:
        train_examples, val_examples = all_examples
        train_bi_labels, val_bi_labels = all_bi_labels

        return train_examples, val_examples, train_bi_labels, val_bi_labels

    train_examples, val_examples, test_examples = all_examples
    train_bi_labels, val_bi_labels, test_bi_labels = all_bi_labels

    return train_examples, val_examples, test_examples, train_bi_labels, val_bi_labels, test_bi_labels


def load_fully_supervised_tv_data(data_id, channels: Union[int, list, np.ndarray] = None, main_data_path = None):
    train_examples, val_examples, train_bi_labels, val_bi_labels = (
        load_fully_supervised_origin_data(data_id, channels=channels, include_test=False, main_data_path=main_data_path))

    tv_examples = np.concatenate((train_examples, val_examples))
    tv_labels = np.concatenate((train_bi_labels, val_bi_labels))

    return tv_examples, tv_labels


def to_seanet_format(tvt_id, base_save_path, positive_examples_, negative_examples_, main_save_path, overwrite=False):

    if tvt_id == 'val':
        tvt_id = 'valid'

    assert positive_examples_.dtype == negative_examples_.dtype == np.float32
    assert positive_examples_.ndim in (2, 3)
    assert positive_examples_.ndim == negative_examples_.ndim
    assert positive_examples_.shape[1:] == negative_examples_.shape[1:]

    if positive_examples_.ndim == 3:
        positive_examples, negative_examples = positive_examples_, negative_examples_
    else:
        positive_examples = positive_examples_[:, np.newaxis, :]
        negative_examples = negative_examples_[:, np.newaxis, :]

    save_path = main_save_path / base_save_path.lstrip("/")
    DirProcessor.create_dir(save_path)

    preprocessed_filename_positive = tvt_id + '-ieds-' + str(int(positive_examples.shape[0])) + '_' + str(
        int(positive_examples.shape[1])) + '.pkl'
    preprocessed_filepath_positive = os.path.join(save_path, preprocessed_filename_positive)
    if overwrite or (not os.path.exists(preprocessed_filepath_positive)):
        positive_examples.tofile(preprocessed_filepath_positive)

    preprocessed_filename_negative = tvt_id + '-nonieds-' + str(int(negative_examples.shape[0])) + '_' + str(
        int(negative_examples.shape[1])) + '.pkl'
    preprocessed_filepath_negative = os.path.join(save_path, preprocessed_filename_negative)
    if overwrite or (not os.path.exists(preprocessed_filepath_negative)):
        negative_examples.tofile(preprocessed_filepath_negative)


def load_seanet_format(tvt_id, base_save_path, main_save_path):

    if tvt_id == 'val':
        tvt_id = 'valid'

    save_path = os.path.join(main_save_path, base_save_path.lstrip("/"))

    p_pfx, n_pfx = f'{tvt_id}-ieds-', f'{tvt_id}-nonieds-'
    p_fnames = [fname for fname in os.listdir(save_path) if fname.startswith(p_pfx) and fname.endswith('.pkl')]
    n_fnames = [fname for fname in os.listdir(save_path) if fname.startswith(n_pfx) and fname.endswith('.pkl')]
    assert len(p_fnames) == len(n_fnames) == 1
    p_fname, n_fname = p_fnames[0], n_fnames[0]

    n_p, n_n = int(p_fname.split('_')[0][len(p_pfx):]), int(n_fname.split('_')[0][len(n_pfx):])
    n_channels = int(p_fname.split('_')[1][:-len('.pkl')])

    positive_examples = np.fromfile(os.path.join(save_path, p_fname), dtype=np.float32).reshape((n_p, n_channels, -1))
    negative_examples = np.fromfile(os.path.join(save_path, n_fname), dtype=np.float32).reshape((n_n, n_channels, -1))

    return positive_examples, negative_examples

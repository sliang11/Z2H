# coding = utf-8

import json
import subprocess
# import logging
# import time
import itertools
# import shutil
import os
import argparse
from util.file_util import FileWriter, DirProcessor
from util.seeg_file_util import n_channels_selector, load_seanet_format
from pathlib import Path
from deep_learning.SEANet.util.conf import Conf
from deep_learning.SEANet.util.train import EEGTrainable
from deep_learning.SEANet.util.evaluate import Fbeta
import numpy as np
from util.pytorch_util import set_seed


def train(ied_net_name, data_id, channel_id, dl_data_path, dl_result_path):
    assert ied_net_name in ['AIED', 'iEDeal']

    num_samples_methods = {
        'none',  # use all data
        # 'balanced',   # randomly sample balanced data
        # 'cvpr19',     # randomly sample effecient (defined in cvpr19) data
    }

    sample_methods = {
        'random',
        # 'latent',
    }

    sample_epochs = {
        1,
    }

    betas = {
        # 0.5,
        1.,
        # 2.
    }


    if ied_net_name == 'iEDeal':
        model_names = {
            'resnet',
        }
        loss_functions = {
            # 'mse',
            'ce',  # cross-entropy
            # 'wce',   # weighted cross-entropy
            # 'sf',   # surrogate f1
            'sasu'  # surrogate f1 with samples
        }
    elif ied_net_name == 'AIED':
        model_names = {
            'res1d18',
        }
        loss_functions = {
            'ce',  # cross-entropy
        }
    else:
        raise NotImplementedError
    seperator = '-'

    conf_template_filepath = Path(__file__).resolve().parent / 'conf_templates' / f'template_{ied_net_name}.json'
    with open(conf_template_filepath, 'r') as fin:
        conf_template = json.load(fin)

    dataset_tp_filepaths = None
    dataset_tn_filepaths = None
    dataset_vp_filepaths = None
    dataset_vn_filepaths = None

    for sample_data_filename in os.listdir(dl_data_path):
        sample_data_filepath = os.path.join(
            dl_data_path, sample_data_filename)

        if sample_data_filename.startswith('train-ieds'):
            dataset_tp_filepaths = sample_data_filepath
        elif sample_data_filename.startswith('train-nonieds'):
            dataset_tn_filepaths = sample_data_filepath
        elif sample_data_filename.startswith('valid-ieds'):
            dataset_vp_filepaths = sample_data_filepath
        elif sample_data_filename.startswith('valid-nonieds'):
            dataset_vn_filepaths = sample_data_filepath

    for model_name, loss_function, sample_method, num_samples_method, sample_epoch, beta in itertools.product(
        model_names, loss_functions, sample_methods, num_samples_methods, sample_epochs, betas
    ):

        tune_foldername = seperator.join([str(_) for _ in [
            model_name, loss_function, sample_method, num_samples_method, sample_epoch, beta]])
        tune_folderpath = os.path.join(dl_result_path, tune_foldername)
        tmpdump_filepath = os.path.join(tune_folderpath, f'info.json')

        tmpdump = {}
        for key, value in conf_template.items():
            tmpdump[key] = value

        tmpdump['data_id'] = data_id
        tmpdump['conf_template_filepath'] = tmpdump_filepath
        tmpdump['dataset_tp_filepaths'] = dataset_tp_filepaths
        tmpdump['dataset_tn_filepaths'] = dataset_tn_filepaths
        tmpdump['dataset_vp_filepaths'] = dataset_vp_filepaths
        tmpdump['dataset_vn_filepaths'] = dataset_vn_filepaths
        tmpdump['num_input_channels'] = 1 if channel_id != 'all_ch' else n_channels_selector[data_id]
        tmpdump['batch_size'] = 512
        tmpdump['model_name'] = model_name
        tmpdump['loss_function'] = loss_function
        tmpdump['sample_method'] = sample_method
        tmpdump['num_samples_method'] = num_samples_method
        tmpdump['sample_epoch'] = sample_epoch
        tmpdump['beta'] = beta

        if loss_function == 'sasu':
            tmpdump['warmup'] = False
            tmpdump['imbalance_expanding'] = 'none'


        with open(tmpdump_filepath, 'w') as fout:
            fout.write(json.dumps(
                tmpdump, sort_keys=True, indent=4))

        cmd = f'python {str(Path(__file__).resolve.parent / "tune.py")} {tmpdump_filepath}'
        subprocess.run(cmd, shell=True, capture_output=True, text=True)


class Collector(object):

    def __init__(self, result_path,
                 best_by: str, beta: float = 1, __epsilon=1e-5):

        assert best_by in ('train', 'valid', 'val')
        assert beta > 0

        self.result_path = result_path
        self.best_by = 'train' if best_by == 'train' else 'validation'
        self.beta = beta
        self.__epsilon = __epsilon

    def __fscore(self, pre, rec):
        if pre < self.__epsilon or rec < self.__epsilon:
            return 0

        beta2 = self.beta ** 2
        return (1 + beta2) * pre * rec / (beta2 * pre + rec)

    def __collect_seanet(self):

        assert self.beta > 0

        print(f"self.result_path = {self.result_path}")

        assert os.path.exists(self.result_path)

        self.train_fscore_by_model_full_fname = {}  # model_full_fname (== {setting_path}/XXX.pickle) -> fscore of the last epoch
        self.val_fscore_by_model_full_fname = {}    # model_full_fname -> fscore of the last epoch

        for setting_ in os.listdir(self.result_path):  # setting：ce/sasu
            setting_paths_ = os.path.join(self.result_path, setting_)

            if not os.path.isdir(setting_paths_):
                continue

            for tune in os.listdir(setting_paths_):
                setting_path = os.path.join(setting_paths_, tune)


                if not os.path.isdir(setting_path):
                    continue

                print(f'\n********* {setting_path} *********\n')

                """
                Identify training log file
                """
                print('Identifying training log file.')
                if 'train.log' not in os.listdir(setting_path):
                    continue
                log_full_fname = os.path.join(setting_path, 'train.log')


                """
                See if training was finished.
                """

                finished = False
                with open(log_full_fname, 'r') as fin:
                    for line in fin:
                        if 'training failed' in line or 'training early stopped' in line or 'training finished' in line:
                            finished = True
                            break

                """
                Identify model checkpoint file
                """
                model_full_fname = os.path.join(setting_path, 'model.pickle')
                if not os.path.exists(model_full_fname):

                    model_fnames = [fname for fname in os.listdir(setting_path) if fname.endswith('.pickle')]
                    if len(model_fnames) == 0:
                        print(f"No saved model found. Skipping.")
                        continue

                    assert len(model_fnames) == 1 and finished
                    model_full_fname = os.path.join(setting_path, model_fnames[0])

                """
                Fetch the F-scores (per epoch) for the current setting_path
                """

                train_fscores, val_fscores = [], []
                with open(log_full_fname, 'r') as fin:
                    for line in fin:
                        if 'l=' in line and 'f' in line and (   # This line corresponds to the tvt_id 'train'
                                'pre=' in line or 'p=' in line) and (
                                'rec=' in line or 'r=' in line):
                            segments = line.split(':')[-1].split(', ')

                            loss, _, pre, rec = [float(segments[i].split('=')[1]) for i in range(4)]
                            fscore = self.__fscore(pre, rec)
                            train_fscores.append(fscore)
                        elif 'f' in line and ('pre=' in line or 'p=' in line) and (  # This line corresponds to the tvt_id 'val'/'valid'
                                'rec=' in line or 'r=' in line):
                            segments = line.split(':')[-1].split(', ')

                            fscore, pre, rec = [float(segments[i].split('=')[1]) for i in range(3)]
                            fscore = self.__fscore(pre, rec)
                            val_fscores.append(fscore)

                self.train_fscore_by_model_full_fname[model_full_fname] = train_fscores[-1]
                self.val_fscore_by_model_full_fname[model_full_fname] = val_fscores[-1]

        """
        Get the best setting and the corresponding model
        """

        fscore_by_model_full_fname = self.train_fscore_by_model_full_fname if self.best_by == 'train' else self.val_fscore_by_model_full_fname
        self.best_model_full_fname, self.best_fscore = None, -1.
        for model_full_fname, fscore in fscore_by_model_full_fname.items():
            if fscore > self.best_fscore:
                self.best_model_full_fname, self.best_fscore = model_full_fname, fscore

        print(f'Best model found at {self.best_model_full_fname} with the {self.best_by} fscore {self.best_fscore}.')

    def collect(self):
        self.__collect_seanet()


def test(dl_data_path, dl_result_path):
    raw_pred_save_fname = os.path.join(dl_result_path, f'raw_predictions.pkl')
    prf_save_fname = os.path.join(dl_result_path, f'prediction_prf_with_beta=1.pkl')

    collector = Collector(dl_result_path, 'val')
    collector.collect()

    model_full_fname = collector.best_model_full_fname
    if model_full_fname is None:
        print('No model trained successfully. Skipping.')
        predictions = None
        FileWriter.dump_pickle(predictions, raw_pred_save_fname)
        f, recall, precision = -1, -1, -1
        FileWriter.dump_pickle({'f': f, 'precision': precision, 'recall': recall}, prf_save_fname)

    """
    Do online prediction
    """

    print('Doing online prediction on the test set.')

    conf_fname = os.path.join(os.path.dirname(model_full_fname), 'conf.json')
    conf = Conf(conf_fname, query_only=True)
    conf.setHP('checkpoint_filepath', model_full_fname)

    if conf.getHP('model') == 'transformer':
        conf.setHP('batch_size', 256)
    else:
        conf.setHP('batch_size', 1024)

    p_ex, n_ex = load_seanet_format('test', os.path.basename(dl_data_path), os.path.dirname(dl_data_path))
    examples = np.concatenate((p_ex, n_ex))
    gt_labels = np.concatenate((np.ones(len(p_ex)), np.zeros(len(n_ex))))

    predictor = EEGTrainable(conf, train=False)
    predictions = predictor.predict(examples)

    f, precision, recall = Fbeta(predictions, gt_labels)

    print(f'Testing fscore = {f}')

    """
    Save prediction results
    """

    FileWriter.dump_pickle(predictions, raw_pred_save_fname)
    FileWriter.dump_pickle({'f': f, 'precision': precision, 'recall': recall}, prf_save_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("ied_net_name", type=str)
    parser.add_argument("data_id", type=str)
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument('--rand_seed', type=int, default=0)
    parser.add_argument('--main_dl_data_path', type=str, default=None)
    parser.add_argument('--main_dl_result_path', type=str, default=None)

    args = parser.parse_args()

    channel_id = args.channel if args.channel is not None else 'all_ch'

    main_dl_data_path = args.main_dl_data_path
    if main_dl_data_path is None:
        main_dl_data_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent.parent),
                                         "data", 'Z2H_dl_data')
    dl_data_path = os.path.join(main_dl_data_path, f'{args.data_id}_{channel_id}')

    main_dl_result_path = args.main_dl_result_path
    if main_dl_result_path is None:
        main_dl_result_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent.parent),
                                           "dl_result")

    dl_result_path = os.path.join(main_dl_result_path, args.ied_net_name, f'{args.data_id}_{channel_id}')
    DirProcessor.create_dir(dl_result_path)

    set_seed(args.rand_seed)

    train(args.ied_net_name, args.data_id, channel_id, dl_data_path, dl_result_path)
    test(dl_data_path, dl_result_path)

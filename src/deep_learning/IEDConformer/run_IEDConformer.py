import torch
from util.file_util import DirProcessor, FileReader
import os
import shutil
from deep_learning.IEDConformer.IED_deep_dearner import IEDDeepLearner
from deep_learning.IEDConformer.IEDConformer import IEDConformer
from argparse import ArgumentParser
from pathlib import Path


conf = {
    'ied_net_name': 'IEDConformer',
    'batch_size': 64,
    'n_epochs': 200,
    'optimizer_name': 'Adam',
    'optimizer_args': {'lr': 1e-3},
}


def instantiate_learner_from_conf(
        dl_result_path, dl_data_path, rand_seed):

    ied_net_cls_map = {
        'IEDConformer': IEDConformer,
    }
    optimizer_cls_map = {
        'Adam': torch.optim.Adam
    }
    scheduler_cls_map = {

    }
    criterion_cls_map = {

    }

    batch_size, n_epochs = conf['batch_size'], conf['n_epochs']
    ied_net_cls = ied_net_cls_map[conf['ied_net_name']]

    optimizer_cls = optimizer_cls_map[conf['optimizer_name']]
    optimizer_kwargs = conf['optimizer_args']

    scheduler_cls = scheduler_cls_map[conf['scheduler_name']] if 'scheduler_name' in conf.keys() else None
    scheduler_kwargs = conf['scheduler_args'] if 'scheduler_args' in conf.keys() else {}    # 注意：不是None!
    scheduler_monitor = conf['scheduler_monitor'] if 'scheduler_monitor' in conf.keys() else None

    criterion_cls = criterion_cls_map[conf['criterion_name']] if 'criterion_name' in conf.keys() else None
    criterion_kwargs = conf['criterion_args'] if 'criterion_args' in conf.keys() else {} # 注意：不是None!

    pretrain_kwargs = conf['pretrain_args'] if 'pretrain_args' in conf.keys() else {} # 注意：不是None!

    return IEDDeepLearner(
        dl_result_path, os.path.basename(dl_data_path), os.path.dirname(dl_data_path), rand_seed,
        batch_size, n_epochs,
        ied_net_cls, optimizer_cls, optimizer_kwargs,
        scheduler_cls=scheduler_cls, scheduler_kwargs=scheduler_kwargs, scheduler_monitor=scheduler_monitor,
        criterion_cls=criterion_cls, criterion_kwargs=criterion_kwargs, pretrain_kwargs=pretrain_kwargs
    )


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('data_id', type=str)
    parser.add_argument('--channel', type=int, default=None)
    parser.add_argument('--rand_seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')

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

    dl_result_path = os.path.join(main_dl_result_path, 'IEDConformer', f'{args.data_id}_{channel_id}')
    DirProcessor.create_dir(dl_result_path)


    """
    Define the paths
    """

    channel_id = args.channel if args.channel is not None else 'all_ch'

    best_val_f1_save_fname = os.path.join(dl_result_path, f'best_val_f1.pkl')
    raw_pred_save_fname = os.path.join(dl_result_path, f'raw_predictions.pkl')
    prf_save_fname = os.path.join(dl_result_path, f'prediction_prf_with_beta=1.pkl')

    """
    Do deep learning
    """

    if args.overwrite:
        shutil.rmtree(dl_result_path)

    ied_deep_learner = instantiate_learner_from_conf(
        dl_result_path, dl_data_path, args.rand_seed)

    print('******** Training *********')
    if not os.path.exists(best_val_f1_save_fname):
        ied_deep_learner.train()

    print('******* Testing *********')
    if not (os.path.exists(raw_pred_save_fname) and os.path.exists(prf_save_fname)):
        ied_deep_learner.test()

    print('All done!')

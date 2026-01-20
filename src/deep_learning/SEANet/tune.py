# coding = utf-8

import os
import json
from pathlib import Path
import itertools
from timeit import default_timer as timer

import numpy as np
from deep_learning.SEANet.util.experiment import Experiment
from deep_learning.SEANet.util.defaults import defaults


def finish_check(trial_folderpath, template):
    finished = False

    if not os.path.exists(trial_folderpath):
        os.mkdir(trial_folderpath)
    else:
        for filename in sorted(os.listdir(trial_folderpath)):
            if filename == template['checkpoint_filename']:
                finished = True
                break
            elif filename == template['log_filename']:
                with open(os.path.join(trial_folderpath, filename), 'r') as fin:
                    for line in fin:
                        if 'training failed' in line or 'training early stopped' in line or 'training finished' in line:
                            finished = True
                            break

    return finished


def time_check(start_timer, leaveover_duration, tune_duration):
    if leaveover_duration is None or tune_duration is None:
        return True

    return timer() - start_timer + leaveover_duration < tune_duration


seperator = '-'

hyperparameter_grids = {
    # 'relu_slope': {1e-2},

    # 'lr_mode': {'linear', 'exponentially'},
    'lr_mode': {'linear'},

    # 'lr_max': {5e-4, 7.5e-4, 1e-3, 2e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2e-2, 2.5e-2},
    # 'lr_max': {5e-3, 7.5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2},
    # 'lr_max': {5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 1.5e-2, 2e-2, 2.5e-2, 3e-2, 3.5e-2, 4e-2, 4.5e-2, 5e-2},
    # 'lr_max': {1e-3, 2e-3, 5e-3, 1e-2, 2e-2},
    'lr_max': {1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2},

    'wd_mode': {'linear'},
    # 'wd_cons': {7},
    'wd_max': {1e-2},
    # 'wd_min': {1e-4},
    'wd_min': {1e-3},

    'clip_grad': {'norm'},
    # 'max_norm': {0.1, 0.5, 0.6, 0.7},
    # 'max_norm': {0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2},
    # 'max_norm': {1.4, 1.5, 2},
    'max_norm': {2},

    'warmup_epochs': {3},
    # 'warmup_epochs': {7},

    # 'imbalance_expanding': {'none'},
    'increxp_numnegbase': {2},
    # 'increxp_epochbase': {np.sqrt(2)},
    'increxp_epochbase': {np.sqrt(3)},

    # assume all trials with earlystop
    'earlystop_type': {'mean'},
    'earlystop_target': {'tloss'},
    # 'earlystop_target': {'vfbeta'},
    'early_stop_tracebacks': {10},  # {7},
}

gradclip_default = 'norm'
gradclip_default_max_norm = 1

earlystop_default_type = 'mean'
earlystop_default_target = 'tloss'
earlystop_default_tracebacks = 10  # 3


def tune(tmpdump_filepath):

    # tmpdump_filepath = args.dumppath
    assert os.path.isfile(tmpdump_filepath)

    with open(tmpdump_filepath, 'r') as fin:
        tune_info = json.load(fin)

    tune_folderpath = Path(
        tune_info['conf_template_filepath']).parent.absolute()

    with open(tune_info['conf_template_filepath'], 'r') as fin:
        template = json.load(fin)

    for key, value in defaults[tune_info['model_name']].items():
        template[key] = value

    template['num_input_channels'] = tune_info['num_input_channels']

    template['train_positive_samples'] = tune_info['dataset_tp_filepaths']
    template['train_negative_samples'] = tune_info['dataset_tn_filepaths']
    template['valid_positive_samples'] = tune_info['dataset_vp_filepaths']
    template['valid_negative_samples'] = tune_info['dataset_vn_filepaths']

    template['model_name'] = tune_info['model_name']
    template['loss_function'] = tune_info['loss_function']
    template['sample_method'] = tune_info['sample_method']
    template['num_samples_method'] = tune_info['num_samples_method']
    template['sample_epoch'] = tune_info['sample_epoch']
    template['f_beta'] = tune_info['beta']

    # patient_id = tune_info['patient_id']
    start_timer = timer()

    tune_duration = None
    leaveover_duration = None


    #template['num_input_channels'] = 1

    if 'True' in template['train_positive_samples'].split('/')[-2].split('znorm_True')[0]:
        assert '230302e-b_0008' in template['train_positive_samples'] or '180410A-A_0001' in template['train_positive_samples'] or '231002e-a_0028' in template['train_positive_samples']
        # template['num_input_channels']=5 if '231002e-a_0028' not in template['train_positive_samples'] else 4
        template['dim_series'] = 46
        template['num_resblock'] = 4
        template['dim_latent'] = 256
        template['batch_size'] = 4096
    elif 'TUEV' in template['train_positive_samples']:
        template['dim_series'] = 250
        template['num_resblock'] = 4
        template['dim_latent'] = 256
        template['batch_size'] = 4096
    else:
        # template['num_input_channels']=5
        template['batch_size'] = 512

    if template['loss_function'] == 'sasu':
        if 'warmup' in tune_info:
            template['warmup'] = tune_info['warmup']
        else:
            template['warmup'] = True

        if 'imbalance_expanding' in tune_info:
            template['imbalance_expanding'] = tune_info['imbalance_expanding']
        else:
            template['imbalance_expanding'] = 'exponential'

        template['normalize_type'] = 'none'
        # template['normalize_type'] = 'input'
    else:
        template['warmup'] = False
        template['imbalance_expanding'] = 'none'
        template['normalize_type'] = 'none'

    for lr_mode in hyperparameter_grids['lr_mode']:
        for lr_max in hyperparameter_grids['lr_max']:
            # if template['loss_function'] not in loss2tune_gradclip:
            if template['loss_function'] != 'sasu':
                if time_check(start_timer, leaveover_duration, tune_duration):
                    trial_foldername = seperator.join([str(x) for x in ([
                        lr_mode, lr_max, 'none',
                        earlystop_default_type, earlystop_default_target, earlystop_default_tracebacks
                    ])])
                    trial_folderpath = os.path.join(
                        tune_folderpath, trial_foldername)

                    if not finish_check(trial_folderpath, template):
                        template['lr_mode'] = lr_mode
                        template['lr_max'] = lr_max

                        # template['clip_grad'] = 'none'
                        template['clip_grad'] = gradclip_default
                        template['max_norm'] = gradclip_default_max_norm

                        template['earlystop_type'] = earlystop_default_type
                        template['earlystop_target'] = earlystop_default_target
                        template['early_stop_tracebacks'] = earlystop_default_tracebacks

                        template['checkpoint_folderpath'] = trial_folderpath
                        template['log_filepath'] = os.path.join(
                            trial_folderpath, template['log_filename'])
                        template['conf_filepath'] = os.path.join(
                            trial_folderpath, template['default_conf_filename'])

                        with open(template['conf_filepath'], 'w') as fout:
                            json.dump(template, fout, sort_keys=True, indent=4)

                        experiment = Experiment(template['conf_filepath'])
                        experiment.run()
            else:
                # if template['loss_function'] in loss2tune_gradclip and not template['warmup'] and template['imbalance_expanding'] == 'none':
                if not template['warmup'] and template['imbalance_expanding'] == 'none':
                    if time_check(start_timer, leaveover_duration, tune_duration):
                        trial_foldername = seperator.join([str(x) for x in ([
                            lr_mode, lr_max, gradclip_default, gradclip_default_max_norm,
                            earlystop_default_type, earlystop_default_target, earlystop_default_tracebacks
                        ])])
                        trial_folderpath = os.path.join(
                            tune_folderpath, trial_foldername)

                        if not finish_check(trial_folderpath, template):
                            template['lr_mode'] = lr_mode
                            template['lr_max'] = lr_max

                            template['clip_grad'] = gradclip_default
                            template['max_norm'] = gradclip_default_max_norm

                            template['earlystop_type'] = earlystop_default_type
                            template['earlystop_target'] = earlystop_default_target
                            template['early_stop_tracebacks'] = earlystop_default_tracebacks

                            template['checkpoint_folderpath'] = trial_folderpath
                            template['log_filepath'] = os.path.join(
                                trial_folderpath, template['log_filename'])
                            template['conf_filepath'] = os.path.join(
                                trial_folderpath, template['default_conf_filename'])

                            with open(template['conf_filepath'], 'w') as fout:
                                json.dump(template, fout,
                                          sort_keys=True, indent=4)

                            experiment = Experiment(
                                template['conf_filepath'])
                            experiment.run()
                # elif template['loss_function'] in loss2tune_gradclip and template['warmup'] and template['imbalance_expanding'] == 'none':
                elif template['warmup'] and template['imbalance_expanding'] == 'none':
                    for warmup_epochs in hyperparameter_grids['warmup_epochs']:
                        if time_check(start_timer, leaveover_duration, tune_duration):
                            trial_foldername = seperator.join([str(x) for x in ([
                                lr_mode, lr_max, gradclip_default, gradclip_default_max_norm, warmup_epochs,
                                earlystop_default_type, earlystop_default_target, earlystop_default_tracebacks
                            ])])
                            trial_folderpath = os.path.join(
                                tune_folderpath, trial_foldername)

                            if not finish_check(trial_folderpath, template):
                                template['lr_mode'] = lr_mode
                                template['lr_max'] = lr_max

                                template['clip_grad'] = gradclip_default
                                template['max_norm'] = gradclip_default_max_norm

                                template['warmup_epochs'] = warmup_epochs

                                template['earlystop_type'] = earlystop_default_type
                                template['earlystop_target'] = earlystop_default_target
                                template['early_stop_tracebacks'] = earlystop_default_tracebacks

                                template['checkpoint_folderpath'] = trial_folderpath
                                template['log_filepath'] = os.path.join(
                                    trial_folderpath, template['log_filename'])
                                template['conf_filepath'] = os.path.join(
                                    trial_folderpath, template['default_conf_filename'])

                                with open(template['conf_filepath'], 'w') as fout:
                                    json.dump(template, fout,
                                              sort_keys=True, indent=4)

                                experiment = Experiment(
                                    template['conf_filepath'])
                                experiment.run()
                # elif template['loss_function'] in loss2tune_gradclip and not template['warmup'] and template['imbalance_expanding'] != 'none':
                elif not template['warmup'] and template['imbalance_expanding'] != 'none':
                    for increxp_epochbase, increxp_numnegbase in itertools.product(
                        hyperparameter_grids['increxp_epochbase'], hyperparameter_grids['increxp_numnegbase']
                    ):
                        if time_check(start_timer, leaveover_duration, tune_duration):
                            trial_foldername = seperator.join([str(x) for x in ([
                                lr_mode, lr_max, gradclip_default, gradclip_default_max_norm, increxp_epochbase, increxp_numnegbase,
                                earlystop_default_type, earlystop_default_target, earlystop_default_tracebacks
                            ])])
                            trial_folderpath = os.path.join(
                                tune_folderpath, trial_foldername)

                            if not finish_check(trial_folderpath, template):
                                template['lr_mode'] = lr_mode
                                template['lr_max'] = lr_max

                                template['clip_grad'] = gradclip_default
                                template['max_norm'] = gradclip_default_max_norm

                                template['increxp_epochbase'] = increxp_epochbase
                                template['increxp_numnegbase'] = increxp_numnegbase

                                template['earlystop_type'] = earlystop_default_type
                                template['earlystop_target'] = earlystop_default_target
                                template['early_stop_tracebacks'] = earlystop_default_tracebacks

                                template['checkpoint_folderpath'] = trial_folderpath
                                template['log_filepath'] = os.path.join(
                                    trial_folderpath, template['log_filename'])
                                template['conf_filepath'] = os.path.join(
                                    trial_folderpath, template['default_conf_filename'])

                                with open(template['conf_filepath'], 'w') as fout:
                                    json.dump(template, fout,
                                              sort_keys=True, indent=4)

                                experiment = Experiment(
                                    template['conf_filepath'])
                                experiment.run()
                else:
                    for clip_grad, max_norm, warmup_epochs, increxp_epochbase, increxp_numnegbase, \
                            earlystop_type, earlystop_target, early_stop_tracebacks, \
                            wd_mode, wd_max, wd_min in itertools.product(
                                hyperparameter_grids['clip_grad'], hyperparameter_grids['max_norm'],
                                hyperparameter_grids['warmup_epochs'],
                                hyperparameter_grids['increxp_epochbase'], hyperparameter_grids['increxp_numnegbase'],
                                hyperparameter_grids['earlystop_type'], hyperparameter_grids[
                                    'earlystop_target'], hyperparameter_grids['early_stop_tracebacks'],
                                hyperparameter_grids['wd_mode'], hyperparameter_grids['wd_max'], hyperparameter_grids['wd_min']
                            ):
                        if time_check(start_timer, leaveover_duration, tune_duration):
                            trial_foldername = seperator.join([str(x) for x in ([
                                lr_mode, lr_max, clip_grad, max_norm, warmup_epochs,
                                increxp_epochbase, increxp_numnegbase, earlystop_type, earlystop_target,
                                wd_mode, wd_max, wd_min
                            ])])
                            trial_folderpath = os.path.join(
                                tune_folderpath, trial_foldername)

                            if not finish_check(trial_folderpath, template):
                                template['lr_mode'] = lr_mode
                                template['lr_max'] = lr_max

                                template['wd_mode'] = wd_mode
                                template['wd_max'] = wd_max
                                template['wd_min'] = wd_min

                                template['clip_grad'] = clip_grad
                                template['max_norm'] = max_norm

                                template['warmup_epochs'] = warmup_epochs

                                template['increxp_epochbase'] = increxp_epochbase
                                template['increxp_numnegbase'] = increxp_numnegbase

                                template['earlystop_type'] = earlystop_type
                                template['earlystop_target'] = earlystop_target
                                template['early_stop_tracebacks'] = early_stop_tracebacks
                                # template['earlystop_threshold_vfbeta'] = patient_earlystop_vfbetathresholds[patient_id]

                                template['checkpoint_folderpath'] = trial_folderpath
                                template['log_filepath'] = os.path.join(
                                    trial_folderpath, template['log_filename'])
                                template['conf_filepath'] = os.path.join(
                                    trial_folderpath, template['default_conf_filename'])

                                with open(template['conf_filepath'], 'w') as fout:
                                    json.dump(template, fout,
                                              sort_keys=True, indent=4)

                                experiment = Experiment(
                                    template['conf_filepath'])
                                experiment.run()

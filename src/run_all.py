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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("ied_net_name", type=str,
                        choices=['AIED', 'iEDeal', 'IEDConformer', 'EEGPT', 'NeuroGPT'])
    parser.add_argument("data_id", type=str)
    parser.add_argument('--rand_seed', type=int, default=0)

    args = parser.parse_args()

    z2h_cmd = (f'python {Path(__file__).resolve().parent / "Z2H" / "run_Z2H.py"} {args.data_id}'
               f' --rand_seed {args.rand_seed}')
    subprocess.run(z2h_cmd, shell=True, capture_output=True, text=True)

    if args.ied_net_name in ['AIED', 'iEDeal']:
        dl_cmd = (f'python {Path(__file__).resolve().parent / "deep_learning" / "SEANet" / "run_SEANet.py"}'
                  f' {args.ied_net_name} {args.data_id} --rand_seed {args.rand_seed}')
    elif args.ied_net_name == 'IEDConformer':
        dl_cmd = (f'python {Path(__file__).resolve().parent / "deep_learning" / "IEDConformer" / "run_IEDConformer.py"}'
                  f' {args.data_id} --rand_seed {args.rand_seed}')
    elif args.ied_net_name == 'EEGPT':
        dl_cmd = (f'python {Path(__file__).resolve().parent / "deep_learning" / "EEGPT" / "run_EEGPT.py"}'
                  f' {args.data_id} --rand_seed {args.rand_seed}')
    elif args.ied_net_name == 'NeuroGPT':
        dl_cmd = (f'python {Path(__file__).resolve().parent / "deep_learning" / "NeuroGPT" / "run_NeuroGPT.py"}'
                  f' {args.data_id} --rand_seed {args.rand_seed} --freeze-embedder --freeze-decoder '
                  f'--freeze-unembedder --ft-only_encoder --training-style decoding --num-decoding-classes 2'
                  f'--per-device-training-batch-size 16 --training-steps 20000')


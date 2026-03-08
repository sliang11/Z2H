# Z2H

This is a repository holding the source code, mouse data as well as other supplemental mateirals, intended for the double-blind review of our paper *Zero-to-Hero: Knowledge-guided Unsupervised Active Learning for EEG-based Detection of
Interictal Epileptiform Discharges*, submitted to the VLDB 2026.

## Downloading the Source Code and Mouse Data
Clone this repository to your local machine to download the source code and mouse data.

## Downloading the Pretrained Parameters

Aside from cloning this repository to your local directory, you need to additionally download the pretrained parameters for EEGPT and Neuro-GPT.

- For EEGPT, follow the instructions at <https://github.com/BINE022/EEGPT> to download `eegpt_mcae_58chs_4s_large4E.ckpt`; in the root path of the local repository, put it under `./src/deep_learning/EEGPT`.
- For Neuro-GPT, follow the link <https://drive.google.com/file/d/1_q220i_sFNCqIUFGyFCoGcE4SA5ihBLK/view?usp=sharing> to download `pytorch_model.bin`; in the root path of the local repository, put it under `./src/deep_learning/NeuroGPT`.

## Running the Code
In the root path of the downloaded code, run

```
pip install -e .
python ./src/run_all.py [core_model_name] [data_id]
```

where

- `core_model_name` can be one of `AIED`, `iEDeal`, `IEDConformer`, `EEGPT`, `NeuroGPT`;
- `data_id` can be one of `MO1`, `MO2`, `MO3`.

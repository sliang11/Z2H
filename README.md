# Z2H

This is a repository holding the source code and mouse data, intended for the single-blind review of our paper "Zero-to-Hero: A Knowledge-guided Unsupervised Active Learning Pipeline for EEG-based Detection of Interictal Epileptiform Discharges [Scalable Data Science]", submitted to the VLDB 2026.

## Supplemental Materials

Due to page limits, we were unable to include all experimental details and raw experimental results in our submitted manuscript. While we fully understand and respect the reviewers' right to judge our work solely on the basis of the submitted manuscript, to help better understand our work, we invite the reviewers to refer to ***an extended version of our paper in the file `Z2H_extended_version.pdf`***, as well as the raw experimental results in `raw_results.xlsx`.


## Downloading the pretrained parameters

Aside from cloning this repository to your local directory, you need to additionally download the pretrained parameters for EEGPT and Neuro-GPT

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

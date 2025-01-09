# Z2H

This is an anonymous repository holding the source code and mice data, intended for the double-blind review of  our paper "Zero-to-Hero: Knowledge-guided Active Learning for Interictal Epileptiform Discharges Detection in Unlabeled EEG Data", submitted to the KDD'25 ADS track.

## Downloading the mice data
To download the mice data, follow this [link](https://drive.google.com/file/d/1Lzo7PLL6PrHqdwgvzbmkK4TpBiIyDjoQ/view?usp=sharing).

## Running the code
Suppose you have unzipped the mice data into the same directory as the code, follow the steps below to run it.

### 1. Pre-compute the Nearest Neighbor (NN) distance matrices
As a warmup routine, run the following to pre-compute the NN distances, which will be used in both Modules M1 and M2.

```
cd $code_dir
python ./Warmup_calc_dist_mat.py $data_id
```

Here, `$data_id` can be one of `MO1`, `MO2`, or `MO3`.

### 2. Execute Module M1 (Knowledge-guided Active Initialization)
To run Module M1, use the following.

```
python ./M1_active_initialization.py $data_id
```

### 3. Execute Module M2 (Actively Enhanced PU Transduction) and Knowledge-guided Training Examples Filtering in Module M3 (Knowledge-guided Robust Deep Model Training)
To run Module M2 to get a fully labeled training set, as well as filtering likely mislabeled examples in them using the KTEF routine in M3, use the following.

```
python ./M2_active_pu_transduction.py $data_id 0.1 0.7 0 amp_dist
```

This will lead to the filtered dataset stored at `./result/filtered_$data_id.pkl` by default, which can be loaded via the following Python code.

```
import pickle

data_id = 'MO1' # or 'MO2', 'MO3'

fname = f'./result/filtered_{data_id}.pkl'
with open(fname, 'rb') as f:
  fit_examples, fit_labels, val_examples, val_labels = pickle.load(fname)
```

in which `fit_examples` and `fit_labels` are for model fitting, and `val_examples` and `val_labels` are for validation.

You can then use this filtered dataset to train a core deep model of your choosing. To evaluate your model, use the following Python code to load the test data.

```
import pickle

data_id = 'MO1' # or 'MO2', 'MO3'

fname = f'./data/{data_id}.pkl'
with open(fname, 'rb') as f:
  _, _, _, _, test_examples, test_labels = pickle.load(fname)
```

(Note: In our experiments, we used the original authors' implementations of the iEDeal and WNG-TS-1DCNN core deep models. Due to the lack of explicit permission from the original authors, we are unable to post their implementations here. Please contact the original authors or follow the code links in their papers to gain access to the code.)

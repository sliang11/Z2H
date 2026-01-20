# coding = utf-8

import math

import numpy as np
import torch


class Samples(torch.utils.data.Dataset):
    def __init__(self, samples):
        super(Samples, self).__init__()

        assert type(samples) is torch.Tensor
        assert len(samples.shape) == 3

        self.__len = samples.shape[0]

        self.__samples = samples

        
    def __len__(self):
        return self.__len
    
    
    def __getitem__(self, indices):
        return self.__samples[indices]
    

class SamplesLabels(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        super(SamplesLabels, self).__init__()

        assert type(samples) is torch.Tensor and type(labels) is torch.Tensor
        assert len(samples.shape) == 3 and len(labels.shape) == 1
        assert samples.shape[0] == labels.shape[0]

        self.__len = samples.shape[0]

        self.__samples = samples
        self.__labels = labels

        
    def __len__(self):
        return self.__len
    
    
    def __getitem__(self, indices):
        return self.__samples[indices], self.__labels[indices]
    

class SamplesLabelsWeights(torch.utils.data.Dataset):
    def __init__(self, samples, labels = None, weights = None):
        super(SamplesLabelsWeights, self).__init__()
        
        assert type(samples) is torch.Tensor and len(samples.shape) == 3
        
        self.__len = samples.shape[0]

        if labels is not None:
            assert type(labels) is torch.Tensor and len(labels.shape) == 1 and self.__len == labels.shape[0]

            if weights is not None:
                assert type(weights) is torch.Tensor and len(weights.shape) == 1 and self.__len == weights.shape[0]
        else:
            assert weights is None

        self.__samples = samples
        self.__labels = labels
        self.__weights = weights
        self.__none_filling = float('inf')
        
        
    def __len__(self):
        return self.__len
    
    
    def __getitem__(self, indices):
        # TODO verify this in source code
        # assert type(indices) == int

        if self.__labels is not None:
            if self.__weights is not None:
                return self.__samples[indices], self.__labels[indices], self.__weights[indices]
            else:
                return self.__samples[indices], self.__labels[indices], self.__none_filling
        else:
            return self.__samples[indices], self.__none_filling, self.__none_filling
    

def normalize(values: np.ndarray, axis = -1, mu: np.float32 = 0, sigma: np.float32 = 1, local: bool = True, epsilon: np.float32 = 1e-6):
    # TODO implement for common cases
    assert local
    assert axis == -1
    assert len(values.shape) == 3

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            # TODO not taking into consideration the performance
            local_mu = np.mean(values[i, j])
            local_sigma = np.sqrt(np.var(values[i, j]))

            if local_sigma <= epsilon:
                values[i, j] = mu + np.zeros_like(values[i, j])
            else:
                values[i, j] = mu + sigma * (values[i, j] - local_mu) / local_sigma
    
    return values


# credit: https://github.com/gzerveas/mvts_transformer/blob/3f2e378bc77d02e82a44671f20cf15bc7761671a/src/torch.utils.data.Datasets/torch.utils.data.Dataset.py
def collate_superv(data, device, max_len=None):
    """Build mini-batch torch.Tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch torch.Tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch torch.Tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch torch.Tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch torch.Tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch torch.Tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean torch.Tensor, 1 means keep vector at this position, 0 means padding
    """

    # batch_size = len(data)
    # features, labels, IDs = zip(*data)
    features, labels, weights = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D torch.Tensors, i.e. add batch dimension)
    # lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    # if max_len is None:
    #     max_len = max(lengths)
    # X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    # for i in range(batch_size):
    #     end = min(lengths[i], max_len)
    #     X[i, :end, :] = features[i][:end, :]

    X = torch.swapaxes(torch.stack(features, dim=0), 1, 2)
    
    if max_len is None:
        max_len = X.shape[1]
    lengths = [max_len for _ in features]

    if labels is None or labels[0] is None or (type(labels[0]) is float and math.isinf(labels[0])):
        targets = None
    else:
        targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16, device=device),
                                 max_len=max_len)  # (batch_size, padded_length) boolean torch.Tensor, "1" means keep

    if weights is None or weights[0] is None or (type(weights[0]) is float and math.isinf(weights[0])):
        weights = None
    else:
        weights = torch.stack(weights, dim=0)

    # return X, targets, padding_masks, IDs
    return X, targets, weights, padding_masks
    
# credit: https://github.com/gzerveas/mvts_transformer/blob/3f2e378bc77d02e82a44671f20cf15bc7761671a/src/torch.utils.data.Datasets/torch.utils.data.Dataset.py
def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

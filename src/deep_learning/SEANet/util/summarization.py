# coding = utf-8
# credit: https://github.com/tslearn-team/tslearn/blob/42a56cc/tslearn/piecewise/piecewise.py

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import norm


class TransformBaseModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def transform(self, X):
        pass


class PAA(TransformBaseModel):
    def __init__(self, n_segments=1):
        self.n_segments = n_segments

    @staticmethod
    def to_time_series(ts, remove_nans=False):

        ts_out = np.array(ts, copy=True)
        if ts_out.ndim <= 1:
            ts_out = ts_out.reshape((-1, 1))
        if ts_out.dtype != np.float:
            ts_out = ts_out.astype(np.float)
        if remove_nans:
            # qt: is this a bug of nested calling?
            ts_out = ts_out[:PAA.ts_size(ts_out)]
        return ts_out

    @staticmethod
    def ts_size(ts):
        ts_ = PAA.to_time_series(ts)
        sz = ts_.shape[0]
        while sz > 0 and np.all(np.isnan(ts_[sz - 1])):
            sz -= 1
        return sz

    def transform(self, X):
        is_1d = False
        if len(X.shape) == 2:
            is_1d = True
            X = X.reshape(X.shape[0], X.shape[1], 1)

        n_ts, sz, d = X.shape
        X_transformed = np.empty((n_ts, self.n_segments, d))

        for i_ts in range(n_ts):
            sz_segment = PAA.ts_size(X[i_ts]) // self.n_segments

            for i_seg in range(self.n_segments):
                start = i_seg * sz_segment
                end = start + sz_segment
                segment = X[i_ts, start:end, :]
                X_transformed[i_ts, i_seg, :] = segment.mean(axis=0)

        if is_1d:
            X_transformed = X_transformed.squeeze()

        return X_transformed


class SAX(PAA):
    def __init__(self, n_segments=1, max_cardinality=4):
        super().__init__(n_segments=n_segments)

        self.max_cardinality = max_cardinality
        self.alphabet_size_avg = 2 ** max_cardinality

        self.breakpoints_avg_ = SAX.breakpoints(self.alphabet_size_avg)

    @staticmethod
    def breakpoints(n_bins, scale=1., offset=0.):
        return norm.ppf([float(a) / n_bins for a in range(1, n_bins)], scale=scale) + offset

    @staticmethod
    def paa_to_symbols(X_paa, breakpoints):
        alphabet_size = breakpoints.shape[0] + 1
        X_symbols = np.zeros(X_paa.shape, dtype=np.int) - 1

        for idx_bp, bp in enumerate(breakpoints):
            indices = np.logical_and(X_symbols < 0, X_paa < bp)
            X_symbols[indices] = idx_bp

        X_symbols[X_symbols < 0] = alphabet_size - 1

        return X_symbols

    def transform(self, X):
        X_paa = super().transform(X)
        self.breakpoints_avg_ = SAX.breakpoints(self.alphabet_size_avg, np.std(X_paa), np.mean(X_paa))
        
        return SAX.paa_to_symbols(X_paa, self.breakpoints_avg_)


class InvSAX(SAX):
    def __init__(self, n_segments=1, max_cardinality=4):
        super().__init__(n_segments=n_segments, max_cardinality=max_cardinality)

        assert n_segments < 64

    def transform(self, X):
        X_sax = super().transform(X)
        X_inv = np.zeros(
            [X_sax.shape[0], self.max_cardinality], dtype=np.uint64)

        for cardinality in range(self.max_cardinality):
            # e.g., Gray code might be better
            for idx_segment in range(self.n_segments):
                X_inv[:, cardinality] = np.left_shift(X_inv[:, cardinality], 1) + np.bitwise_and(
                    np.right_shift(X_sax[:, idx_segment], self.max_cardinality - cardinality - 1), 1)

        return X_inv

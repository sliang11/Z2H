import numpy as np
import torch
from typing import Union
from Z2H.calc_knn_dists import calc_knn_from_ndarray


class AmpAndDist(object):

    def __init__(self, multi_channel_examples: np.ndarray,
                 knn_k=10000,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        assert multi_channel_examples.ndim == 3
        self.multi_channel_examples = multi_channel_examples
        self.device = device

        self.knn_k = knn_k
        self.knn_dist_mat = self.knn_inds_mat = None

    def calc_amp(self, amp_calculator: str = None, channel: int = None):

        if amp_calculator == 'p2p_single':  # single channel
            examples = self.multi_channel_examples[:, channel, :]
            amp = np.max(examples, axis=1) - np.min(examples, axis=1)
            assert amp.shape == (self.multi_channel_examples.shape[0],)
        elif amp_calculator == 'p2p_add':
            assert channel is None
            p2p_by_ch = self.calc_amp(amp_calculator='p2p_by_ch')
            amp = np.sum(p2p_by_ch, axis=1)
            assert amp.shape == (self.multi_channel_examples.shape[0],)
        elif amp_calculator == 'p2p_norm_add':
            assert channel is None
            norm_p2p_by_ch = self.calc_amp(amp_calculator='norm_p2p_by_ch')
            amp = np.sum(norm_p2p_by_ch, axis=1)
            assert amp.shape == (self.multi_channel_examples.shape[0],)
        elif amp_calculator == 'p2p_max':
            assert channel is None
            p2p_by_ch = self.calc_amp(amp_calculator='p2p_by_ch')
            amp = np.max(p2p_by_ch, axis=1)
            assert amp.shape == (self.multi_channel_examples.shape[0],)
        elif amp_calculator == 'p2p_by_ch':
            assert channel is None
            amp = np.max(self.multi_channel_examples, axis=2) - np.min(self.multi_channel_examples, axis=2)
            assert amp.shape == (self.multi_channel_examples.shape[0], self.multi_channel_examples.shape[1])
        elif amp_calculator == 'norm_p2p_by_ch':
            assert channel is None
            p2p_by_ch = self.calc_amp(amp_calculator='p2p_by_ch')
            p2p_mean_by_ch = self.calc_amp(amp_calculator='p2p_mean_by_ch')
            amp = p2p_by_ch / (p2p_mean_by_ch + 1e-20)[np.newaxis, :]
            assert amp.shape == (self.multi_channel_examples.shape[0], self.multi_channel_examples.shape[1])
        elif amp_calculator == 'p2p_mean_by_ch':
            assert channel is None
            p2p_by_ch = self.calc_amp(amp_calculator='p2p_by_ch')
            amp = np.mean(p2p_by_ch, axis=0)
            assert amp.shape == (self.multi_channel_examples.shape[
                                     1],)
        elif amp_calculator == 'p2p_var_by_ch':
            assert channel is None
            p2p_by_ch = self.calc_amp(amp_calculator='p2p_by_ch')
            amp = np.var(p2p_by_ch, axis=0)
            assert amp.shape == (self.multi_channel_examples.shape[1],)
        elif amp_calculator == 'norm_p2p_var_by_ch':
            assert channel is None
            norm_p2p_by_ch = self.calc_amp(amp_calculator='norm_p2p_by_ch')
            amp = np.var(norm_p2p_by_ch, axis=0)
            assert amp.shape == (self.multi_channel_examples.shape[1],)
        else:
            raise NotImplementedError(f"amp_calculator {amp_calculator} not implemented.")

        return amp

    def calc_dist(self, dist_calculator: str, channels: Union[list, np.ndarray, int] = None, knn_dist_mat=None):

        if not self.channels_unchanged(channels=channels):
            self.calc_knn_from_scratch(channels=channels)

        if dist_calculator == '1nn':
            dist = self.knn_dist_mat[:, 0] if knn_dist_mat is None else knn_dist_mat[:, 0]
        else:
            raise NotImplementedError(f"dist_calculator {dist_calculator} not implemented.")

        assert dist.ndim == 1 and dist.shape[0] == self.multi_channel_examples.shape[0]
        return dist

    def calc_knn_from_scratch(self, channels: Union[list, np.ndarray, int] = None):
        if isinstance(channels, int) or channels is None:
            self.dist_channels = channels
        else:
            self.dist_channels = np.sort(channels)  # we always sort the channels

        self.knn_dist_mat, self.knn_inds_mat = (
            calc_knn_from_ndarray(self.multi_channel_examples, self.dist_channels, k=self.knn_k))

    def channels_unchanged(self, channels: Union[list, np.ndarray, int] = None):
        if self.knn_dist_mat is None:
            assert self.knn_inds_mat is None
            return False
        if isinstance(channels, type(self.dist_channels)):
            return False
        if isinstance(channels, int) or channels is None:
            return channels == self.dist_channels
        return (np.sort(channels) == self.dist_channels).all()

import numpy as np
from Z2H.amp_and_dist import AmpAndDist as AD



# abstract class
class M3(object):

    def __init__(self, multi_channel_examples, M1_queried_inds=None, M2_queried_inds=None, M2_predicted_labels=None, **kwargs):

        if M2_predicted_labels is not None:
            assert len(multi_channel_examples) == len(M2_predicted_labels)

        self.multi_channel_examples = multi_channel_examples
        self.n_examples, self.n_channels, self.ts_len = self.multi_channel_examples.shape

        self.M2_predicted_labels = M2_predicted_labels

        assert not ((M1_queried_inds is None) ^ (M2_queried_inds is None))  # 同时是或者不是None

        if M1_queried_inds is None:
           self.M1_queried_inds = self.M2_queried_inds = self.active_queried_inds = self.inferred_inds = None
           self.n_queried = self.n_inferred = None
        else:
            self.M1_queried_inds = np.array(M1_queried_inds)
            self.M2_queried_inds = np.array(M2_queried_inds)
            self.active_queried_inds = np.concatenate((self.M1_queried_inds, self.M2_queried_inds))
            self.inferred_inds = np.setdiff1d(np.arange(self.n_examples), self.active_queried_inds)
            self.n_queried, self.n_inferred = len(self.active_queried_inds), len(self.inferred_inds)

        self.kwargs = kwargs

    def select_inds(self):
        self.core()
        self.selected_inds = np.sort(self.selected_inds)

    def core(self):
        self.selected_inds = [] # Implement this method!

class M3_No_Filter(M3):
    def __init__(self, multi_channel_examples):
        super().__init__(multi_channel_examples)

    def core(self):
        self.selected_inds = np.arange(self.n_examples)


class M3_Only_Manual(M3):

    def __init__(self, multi_channel_examples, M1_queried_inds, M2_queried_inds):
        assert M1_queried_inds is not None
        assert M2_queried_inds is not None
        super().__init__(multi_channel_examples, M1_queried_inds=M1_queried_inds, M2_queried_inds=M2_queried_inds)

    def core(self):
        self.selected_inds = self.active_queried_inds


class M3_KTEF(M3):

    valid_base_filter_strategies=('amp', 'dist', 'amp_dist')

    def __init__(
            self, multi_channel_examples, M1_best_channel, M1_queried_inds, M2_queried_inds, M2_aie_inds,
            M2_predicted_labels,
            base_filter_strategy: str, amp_calculator: str=None, dist_calculator: str=None,
            amp_filter_th='median', dist_filter_th='median',
            knn_dist_mat=None, knn_inds_mat=None,
    ):
        assert M1_queried_inds is not None
        assert M2_queried_inds is not None
        assert len(np.intersect1d(M2_aie_inds, np.concatenate((M1_queried_inds, M2_queried_inds)))) == 0
        assert len(np.intersect1d(M1_queried_inds, M2_queried_inds)) == 0

        assert base_filter_strategy in self.valid_base_filter_strategies

        if 'amp' in base_filter_strategy:
            assert amp_calculator is not None
        if 'dist' in base_filter_strategy:
            assert dist_calculator is not None

        super().__init__(multi_channel_examples, M1_queried_inds=M1_queried_inds, M2_queried_inds=M2_queried_inds,
                         M2_predicted_labels=M2_predicted_labels)

        self.aie_inds = M2_aie_inds

        self.aie_p_inds = self.aie_inds[M2_predicted_labels[M2_aie_inds] == 1]
        self.aie_n_inds = self.aie_inds[M2_predicted_labels[M2_aie_inds] == 0]

        self.channel = M1_best_channel

        self.base_filter_strategy = base_filter_strategy
        self.amp_calculator = amp_calculator
        self.dist_calculator = dist_calculator
        self.amp_filter_th = amp_filter_th
        self.dist_filter_th = dist_filter_th

        self.knn_dist_mat = knn_dist_mat
        self.knn_inds_mat = knn_inds_mat
        self.ad = AD(multi_channel_examples)

    def select_aie_inds_by_amp(self):
        amp = self.ad.calc_amp(self.amp_calculator, channel=self.channel)

        if self.amp_filter_th == 'median':
            filter_th_val = np.median(amp)
        else:
            raise NotImplementedError()
        inds_high_amp = np.where(amp > filter_th_val)[0]
        inds_low_amp = np.where(amp <= filter_th_val)[0]

        selected_inferred_p = np.intersect1d(inds_high_amp, self.aie_p_inds)
        selected_inferred_n = np.intersect1d(inds_low_amp, self.aie_n_inds)

        return np.concatenate((selected_inferred_p, selected_inferred_n))

    def select_aie_inds_by_dist(self):
        dist = self.ad.calc_dist(self.dist_calculator, knn_dist_mat=self.knn_dist_mat)

        if self.dist_filter_th == 'median':
            filter_th_val = np.median(dist)
        else:
            raise NotImplementedError()
        inds_high_dist = np.where(dist > filter_th_val)[0]
        inds_low_dist = np.where(dist <= filter_th_val)[0]

        selected_inferred_p = np.intersect1d(inds_low_dist, self.aie_p_inds)
        selected_inferred_n = np.intersect1d(inds_high_dist, self.aie_n_inds)

        return np.concatenate((selected_inferred_p, selected_inferred_n))

    def core(self):
        if self.base_filter_strategy == 'amp':
            self.selected_aie_inds = self.select_aie_inds_by_amp()
        elif self.base_filter_strategy == 'dist':
            self.selected_aie_inds = self.select_aie_inds_by_dist()
        elif self.base_filter_strategy == 'amp_dist':
            self.selected_aie_inds = np.intersect1d(
                self.select_aie_inds_by_amp(), self.select_aie_inds_by_dist()
            )
        else:
            raise NotImplementedError()

        self.selected_inds = np.concatenate((self.active_queried_inds, self.selected_aie_inds))

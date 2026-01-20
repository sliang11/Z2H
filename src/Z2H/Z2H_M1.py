import numpy as np
from typing import Union
from time import perf_counter
from Z2H.amp_and_dist import AmpAndDist as AD

# abstract class
class M1(object):

    def __init__(self, multi_channel_examples: np.ndarray, real_labels: np.ndarray, base_sampling_strategy: str, **kwargs):

        assert multi_channel_examples.ndim == 3 and real_labels.ndim == 1
        assert len(multi_channel_examples) == len(real_labels)

        self.multi_channel_examples = multi_channel_examples
        self.real_labels = real_labels
        self.n_examples = multi_channel_examples.shape[0]

        self.base_sampling_strategy = base_sampling_strategy

        self.kwargs = kwargs

        self.response_times = []
        self.tic = None  # Used to obtain user interaction response time

        self.queried_inds = []
        self.label_by_queried_ind = []
        self.remaining_queries = None

    def one_query(self, ind):
        if self.tic is not None:
            self.response_times.append(perf_counter() - self.tic)

        assert ind not in self.queried_inds
        self.queried_inds.append(ind)
        self.label_by_queried_ind.append(self.real_labels[ind])
        self.remaining_queries -= 1

        self.tic = perf_counter()

    def check_n_queries(self, n_queries: int = None):
        if n_queries is None:
            n_queries = self.n_examples
        assert n_queries >= 1 and n_queries <= self.n_examples
        self.remaining_queries = n_queries

        return n_queries

    def active_queries(self, n_queries: int = None):
        n_queries = self.check_n_queries(n_queries=n_queries)
        self.core(n_queries)
        self.queried_inds = np.array(self.queried_inds)
        self.label_by_queried_ind = np.array(self.label_by_queried_ind)

        assert len(self.queried_inds) == n_queries - self.remaining_queries
        assert len(np.unique(self.queried_inds)) == len(self.queried_inds)
        assert len(self.queried_inds) == len(self.label_by_queried_ind)

    def core(self, n_queries: int):
        raise NotImplementedError()  # core method to implement


class M1_Z2H(M1):

    valid_single_channel_amp_calculators = ('p2p_single',)
    valid_multi_channel_amp_calculators = ('p2p_add', 'p2p_norm_add', 'p2p_max')
    valid_single_channel_dist_calculators = ('1nn',)
    valid_multi_channel_dist_calculators = ('1nn',)
    valid_base_sampling_stategies = ('amp', 'dist', 'amp_dist', 'dist_amp')

    def __init__(self, data_id, channel: Union[None, int],

                 multi_channel_examples: np.ndarray, real_labels: np.ndarray,

                 base_sampling_strategy: str,
                 amp_calculator: str = None, dist_calculator: str = None,
                 filter_threshold: str = 'median', adaptive_filter: str = None,

                 channel_ranker: str = None, ch_select_warmup: str = None,

                 **kwargs):

        super().__init__(multi_channel_examples, real_labels, base_sampling_strategy, **kwargs)

        assert len(base_sampling_strategy.split('_')) in (1, 2)
        assert base_sampling_strategy in self.valid_base_sampling_stategies

        if channel is not None:
            assert isinstance(channel, int)

        if isinstance(channel, int):

            self.ch_usage = 'only_one_avail'  # only one channel is given

            channel_ranker = None
            ch_select_warmup = None
            assert amp_calculator in self.valid_single_channel_amp_calculators
            assert dist_calculator in self.valid_single_channel_dist_calculators
        elif channel_ranker is None:

            self.ch_usage = 'all'   # use all channels

            ch_select_warmup = None
            assert amp_calculator in self.valid_multi_channel_amp_calculators
            assert dist_calculator in self.valid_multi_channel_dist_calculators
        else:

            self.ch_usage = 'one_of_all'    # use only one of all channels

            assert ((amp_calculator in self.valid_single_channel_amp_calculators) or
                    (dist_calculator in self.valid_single_channel_dist_calculators))

        ch_select_by_usage = {
            'only_one_avail': False,
            'all': False,
            'one_of_all': True,
        }
        self.do_ch_select = ch_select_by_usage[self.ch_usage]

        self.n_channels = None if self.ch_usage == 'only_one_avail' else multi_channel_examples.shape[1]

        self.data_id = data_id
        self.best_channel = channel

        self.dist_channel_id = -1
        self.ad = AD(multi_channel_examples)

        # self.base_sampling_strategy = base_sampling_strategy
        self.primary_indicator = self.base_sampling_strategy.split('_')[0]
        self.secondary_indicator = self.base_sampling_strategy.split('_')[1] if len(self.base_sampling_strategy.split('_')) > 1 else None

        factor_by_indicator = {
            'amp': -1.,
            'dist': 1.,
        }
        self.argsort_factor = factor_by_indicator[self.primary_indicator]
        self.filter_factor = factor_by_indicator[self.secondary_indicator] if self.secondary_indicator is not None else None

        self.amp_calculator = amp_calculator
        self.dist_calculator = dist_calculator
        self.filter_threshold = filter_threshold
        self.adaptive_filter = adaptive_filter

        self.channel_ranker = channel_ranker
        self.ch_select_warmup = ch_select_warmup

        self.amp_by_calculator = {}
        self.dist_by_calculator = {}


    def rank_channels(self):
        assert self.channel_ranker is not None

        if self.channel_ranker == 'largest_amp':
            p2p_by_ch = self.calc_amp(amp_calculator='p2p_by_ch')
            max_p2p_by_ch = np.max(p2p_by_ch, axis=0)
            assert max_p2p_by_ch.shape == (self.n_channels,)
            self.ch_order = np.argsort(-max_p2p_by_ch)
        elif self.channel_ranker == 'largest_norm_amp':
            norm_p2p_by_ch = self.calc_amp(amp_calculator='norm_p2p_by_ch')
            max_norm_p2p_by_ch = np.max(norm_p2p_by_ch, axis=0)
            assert max_norm_p2p_by_ch.shape == (self.n_channels,)
            self.ch_order = np.argsort(-max_norm_p2p_by_ch)
        elif self.channel_ranker == 'largest_amp_var':
            p2p_var_by_ch = self.calc_amp(amp_calculator='p2p_var_by_ch')
            assert p2p_var_by_ch.shape == (self.n_channels,)
            self.ch_order = np.argsort(-p2p_var_by_ch)
        elif self.channel_ranker == 'largest_norm_amp_var':
            norm_p2p_var_by_ch = self.calc_amp(amp_calculator='norm_p2p_var_by_ch')
            assert norm_p2p_var_by_ch.shape == (self.n_channels,)
            self.ch_order = np.argsort(-norm_p2p_var_by_ch)
        else:
            raise NotImplementedError(f'channel_ranker {self.channel_ranker} not implemented.')

    def calc_amp(self, amp_calculator: str = None, channel: int = None):


        channel_id = 'all_ch' if channel is None else str(channel)
        calculator_with_ch = f'{amp_calculator}_{channel_id}' if self.ch_usage != 'only_one_avail' else amp_calculator

        if amp_calculator is None:
            amp_calculator = self.amp_calculator
        if amp_calculator in self.amp_by_calculator.keys():
            return self.amp_by_calculator[calculator_with_ch]

        amp = self.ad.calc_amp(amp_calculator, channel=channel)
        if amp_calculator in (list(self.valid_single_channel_amp_calculators) + list(self.valid_multi_channel_amp_calculators)):
            assert amp.ndim == 1

        self.amp_by_calculator[calculator_with_ch] = amp
        return amp

    def calc_dist(self, dist_calculator: str = None):

        if dist_calculator is None:
            dist_calculator = self.dist_calculator
        if dist_calculator in self.dist_by_calculator.keys():
            return self.dist_by_calculator[dist_calculator]

        dist = self.ad.calc_dist(dist_calculator, self.best_channel)
        self.dist_by_calculator[dist_calculator] = dist
        return dist

    def get_indicator_values(self, indicator, channel: int = None):
        if indicator is None:
            return None
        if indicator == 'amp':
            return self.calc_amp(channel=channel)
        if indicator == 'dist':
            return self.calc_dist()  # 不需要channel
        raise NotImplementedError()

    def get_primary_query_order(self, primary_indicator_vals_by_ex):
        assert primary_indicator_vals_by_ex is not None
        assert primary_indicator_vals_by_ex.ndim == 1 and len(primary_indicator_vals_by_ex) == self.n_examples
        order = np.argsort(self.argsort_factor * primary_indicator_vals_by_ex)
        return order

    def filter_by_secondary(self, secondary_indicator_vals_by_ex):

        if secondary_indicator_vals_by_ex is None:
            return np.arange(self.n_examples)
        assert secondary_indicator_vals_by_ex.ndim == 1 and len(secondary_indicator_vals_by_ex) == self.n_examples

        if self.filter_threshold == 'median':
            filter_th_val = np.median(secondary_indicator_vals_by_ex)
        else:
            raise NotImplementedError()

        inds_to_keep = np.where(self.filter_factor * (secondary_indicator_vals_by_ex - filter_th_val) < 0)[0]
        return inds_to_keep

    def get_channelwise_query_order(self, channel):
        primary_indi_vals = self.get_indicator_values(self.primary_indicator, channel=channel)
        secondary_indi_vals = self.get_indicator_values(self.secondary_indicator, channel=channel)
        primary_order = self.get_primary_query_order(primary_indi_vals)
        inds_to_keep = np.setdiff1d(self.filter_by_secondary(secondary_indi_vals), self.queried_inds)
        return primary_order[np.isin(primary_order, inds_to_keep)]

    def warmup(self, n_queries):

        if self.do_ch_select:
            assert self.best_channel is None
            self.rank_channels()
            self.best_channel = int(self.ch_order[0])

        if self.ch_select_warmup is None:
            self.query_order = self.get_channelwise_query_order(self.best_channel)
        elif self.ch_select_warmup == 'first_p':
            offset = 0
            while self.remaining_queries:
                self.best_channel = int(self.ch_order[offset])
                # exec(gen_cmd_print_variables('self.best_channel'))
                self.query_order = self.get_channelwise_query_order(self.best_channel)
                self.one_query(self.query_order[0])
                if self.real_labels[self.queried_inds[-1]] == 1:
                    assert self.query_order[1] not in self.queried_inds
                    self.query_order = self.query_order[1:]  # 第0个已经被query过
                    break
                offset = (offset + 1) % self.n_channels
        else:
            raise NotImplementedError()

    def core(self, n_queries):

        self.warmup(n_queries)
        offset = 0
        while self.remaining_queries:
            if self.adaptive_filter is None:
                self.one_query(self.query_order[offset])
                offset += 1
                if offset >= len(self.query_order):
                    return
            elif self.adaptive_filter == 'knn':
                n_neighbors = self.kwargs.get('af_knn_k')
                assert n_neighbors is not None

                if self.ad.knn_dist_mat is None:
                    self.ad.calc_knn_from_scratch(channels=self.best_channel)

                while self.query_order[offset] in self.ad.knn_inds_mat[self.queried_inds, :n_neighbors]:
                    offset += 1
                    if offset >= len(self.query_order):
                        return
                self.one_query(self.query_order[offset])
                offset += 1
                if offset >= len(self.query_order):
                    return
            else:
                raise NotImplementedError()

class M1_Random(M1):

    """
    random sampling
    """

    def __init__(self, multi_channel_examples: np.ndarray, real_labels: np.ndarray):

        super().__init__(multi_channel_examples, real_labels, 'random')

    def core(self, n_queries):
        selected_inds = np.random.choice(self.n_examples, n_queries, replace=False)
        for ind in selected_inds:
            self.one_query(ind)











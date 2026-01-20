# coding = utf-8

from timeit import default_timer as timer

import numpy as np
from sklearn.cluster import AgglomerativeClustering as AggCluster

from deep_learning.SEANet.util.conf import Conf
from deep_learning.SEANet.util.summarization import InvSAX


class Sampler:
    def __init__(self, conf: Conf, logger, instances=None):
        self.__conf = conf
        self.__logger = logger
        self.sample_method_str = self.__conf.getHP('sample_method')

        assert instances is not None and type(
            instances) is np.ndarray and len(instances.shape) > 1
        self.num_instances = instances.shape[0]

        self.instances = instances
        if len(self.instances.shape) > 2:
            self.instances = self.instances.reshape([self.instances.shape[0], -1])

        if self.sample_method_str == 'seasam':
            n_segment = self.__conf.getHP('seasam_n_segment')
            max_cardinality = self.__conf.getHP('seasam_max_cardinality')

            invsax = InvSAX(n_segment, max_cardinality).transform(
                self.instances)

            dtype_keys = [str(x) for x in range(max_cardinality)]
            dtype = [(x, np.uint64) for x in dtype_keys]
            structured_invsax = np.asarray(
                list(zip(*np.transpose(invsax, (1, 0)))), dtype=dtype)  # ugly but effective
            self.sorted_indices = np.argsort(
                structured_invsax, order=dtype_keys)
        elif self.sample_method_str == 'latent':
            pass
        elif self.sample_method_str == 'random':
            pass

    def sample(self, n_samples, candidates: np.ndarray = None):
        if self.sample_method_str == 'seasam':
            if n_samples == 1:
                return np.array([np.random.randint(self.num_instances)])
            elif n_samples <= self.num_instances / 2:
                sample_step = int(np.floor(self.num_instances / n_samples))
                rand_offset = np.random.randint(sample_step)

                sampled_indices = np.arange(
                    rand_offset, self.num_instances, sample_step)
                self.__logger.info(
                    'seasam sampled with {:d} + i * {:d}'.format(rand_offset, sample_step))

                if len(sampled_indices) > n_samples:
                    return np.random.choice(sampled_indices, size=n_samples, replace=False)
                elif len(sampled_indices) < n_samples:
                    raise ValueError('insufficient samples {:d} / {:d} < {:d} '.format(
                        len(sampled_indices), self.num_instances, n_samples))
                else:
                    return sampled_indices
            else:
                sample_step = 2
                rand_offset = np.random.randint(sample_step)

                sampled_indices = np.arange(
                    rand_offset, self.num_instances, sample_step)
                self.__logger.info(
                    'seasam sampled with {:d} + i * {:d}'.format(rand_offset, sample_step))

                return np.concatenate((sampled_indices,
                                      np.random.choice(np.arange(1 - rand_offset, self.num_instances, sample_step),
                                                       size=n_samples - len(sampled_indices), replace=False)))
        elif self.sample_method_str == 'latent':
            assert candidates is not None and len(candidates.shape) == 2
            n_clusters = n_samples  # without grouping

            start = timer()

            # affinitystr or callable, default=’euclidean’. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
            # linkage{‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
            clustering = AggCluster(n_clusters=n_clusters, affinity='euclidean', compute_full_tree=False,
                                    linkage='complete', distance_threshold=None, compute_distances=False).fit(candidates)

            self.__logger.info(
                'cluster {:d}/{:d} in {:.3f}s'.format(n_samples, candidates.shape[0], timer() - start))

            reverse_index = {}
            for sample_id, cluster_id in enumerate(clustering.labels_):
                if cluster_id not in reverse_index:
                    reverse_index[cluster_id] = [sample_id]
                else:
                    reverse_index[cluster_id].append(sample_id)

            n_sample_each = 1
            return np.squeeze(np.asarray([np.random.choice(reverse_index[cluster_id], size=n_sample_each, replace=False, p=None) for cluster_id in range(n_clusters)]))
        elif self.sample_method_str == 'random':
            return np.random.choice(self.num_instances, size=n_samples, replace=False)

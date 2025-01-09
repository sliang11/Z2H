import ctypes
import numpy as np
import subprocess
import os


class PUStoppingCriteria(object):

    def __init__(self, num_train: int, num_p_labeled: int, min_num_p=None, max_num_p=None,
                 sc_cpp_fname="pu_stopping_criteria.cpp"):
        self.num_train = num_train
        self.num_p_labeled = num_p_labeled
        self.min_num_p = num_p_labeled + 1 if min_num_p is None else min_num_p
        self.max_num_p = num_train - 1 if max_num_p is None else max_num_p

        self.sc_cpp_fname = sc_cpp_fname
        self.sc_lib_fname = '_' + self.sc_cpp_fname.split(".")[0] + ".so"
        if not os.path.exists:
            subprocess.run(f"g++ -shared -o {self.sc_lib_fname} -fPIC {self.sc_cpp_fname}", shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.sc_lib = ctypes.CDLL(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + self.sc_lib_fname)

    def sc_rw(self, min_nn_dists):
        self.sc_lib.sc_RW.argtypes = (
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        self.sc_lib.sc_RW.restype = ctypes.c_int

        assert len(min_nn_dists) == self.num_train - self.num_p_labeled

        min_nn_dists_c = (ctypes.c_double * len(min_nn_dists))(*list(min_nn_dists))
        pre_num_p = self.sc_lib.sc_RW(
            min_nn_dists_c, self.min_num_p, self.max_num_p, self.num_train, self.num_p_labeled)

        return pre_num_p

    def sc_bhrk(self, tss_, ranked_inds_, cardinality: int):
        self.sc_lib.sc_BHRK.argtypes = (
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        )
        self.sc_lib.sc_BHRK.restype = ctypes.c_int

        assert tss_.shape[0] == self.num_train
        assert len(ranked_inds_) == self.num_train

        ts_len = tss_.shape[1]

        tss_flattened = list(tss_.flatten())
        ranked_inds = list(ranked_inds_.astype(int))
        hypo_seq = list(np.empty(ts_len, dtype=int))
        next_seq = list(np.empty(ts_len, dtype=int))

        tss_c = (ctypes.c_double * len(tss_flattened))(*tss_flattened)
        ranked_inds_c = (ctypes.c_int * self.num_train)(*ranked_inds)
        hypo_seq_c = (ctypes.c_int * ts_len)(*hypo_seq)
        next_seq_c = (ctypes.c_int * ts_len)(*next_seq)

        pre_num_p = self.sc_lib.sc_BHRK(
            tss_c, ranked_inds_c, hypo_seq_c, next_seq_c,
            self.min_num_p, self.max_num_p, self.num_train, self.num_p_labeled, ts_len, cardinality
        )

        return pre_num_p

    def sc_gbtrm(self, min_nn_dists, beta):
        self.sc_lib.sc_GBTRM.argtypes = (
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_double)
        self.sc_lib.sc_GBTRM.restype = ctypes.POINTER(ctypes.c_int)

        assert len(min_nn_dists) == self.num_train - self.num_p_labeled

        min_nn_dists_c = (ctypes.c_double * len(min_nn_dists))(*list(min_nn_dists))
        pre_num_ps_c = (ctypes.c_int * 5)(*list(np.empty(5, dtype=int)))
        beta_c = ctypes.c_double(beta)

        self.sc_lib.sc_GBTRM(
            pre_num_ps_c, min_nn_dists_c, self.min_num_p, self.max_num_p, self.num_train, self.num_p_labeled, beta_c,
        )  # this just returns a pointer
        pre_num_ps = [pre_num_ps_c[i] for i in range(5)]  # convert pointer to Python list

        pre_num_ps_ = np.array(pre_num_ps)
        assert (pre_num_ps_ > 0).all() or len(np.unique(pre_num_ps_)) == 1

        return pre_num_ps
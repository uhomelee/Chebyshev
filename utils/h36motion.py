from torch.utils.data import Dataset
import numpy as np
from utils import data_utils


class H36motion(Dataset):

    def __init__(self, path_to_data, actions, input_n=10, output_n=10, split=0, sample_rate=2, data_mean=0,
                 data_std=0):
        """
        read h36m data to get the dct coefficients.
        :param path_to_data:
        :param actions: actions to read
        :param input_n: past frame length
        :param output_n: future frame length
        :param dct_n: number of dct coeff. used
        :param split: 0 train, 1 test, 2 validation
        :param sample_rate: 2
        :param data_mean: mean of expmap
        :param data_std: standard deviation of expmap
        """

        self.path_to_data = path_to_data
        self.split = split
        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])

        acts = data_utils.define_actions(actions)

        # subs = np.array([[1], [5], [11]])
        # acts = ['walking']

        subjs = subs[split]
        all_seqs, dim_ignore, dim_use, data_mean, data_std = data_utils.load_data(path_to_data, subjs, acts,
                                                                                  sample_rate,
                                                                                  input_n + output_n,
                                                                                  data_mean=data_mean,
                                                                                  data_std=data_std,
                                                                                  input_n=input_n)

        self.data_mean = data_mean
        self.data_std = data_std

        # first 6 elements are global translation and global rotation
        dim_used = dim_use[6:]
        self.all_seqs = all_seqs
        self.dim_used = dim_used

        all_seqs = all_seqs[:, :, dim_used]
        all_seqs = all_seqs.transpose(0, 2, 1)
        all_seqs = all_seqs.reshape(-1, input_n + output_n)
        all_seqs = all_seqs.transpose()


        # padding the observed sequence so that it has the same length as observed + future sequence
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)

        input_chebyshev_coef = data_utils.get_chebyshev_coef(input_n + output_n, all_seqs[i_idx, :])
        input_chebyshev_coef = input_chebyshev_coef.transpose().reshape([-1, len(dim_used), input_n + output_n])

        output_chebyshev_coef = data_utils.get_chebyshev_coef(input_n + output_n, all_seqs)
        output_chebyshev_coef = output_chebyshev_coef.transpose().reshape([-1, len(dim_used), input_n + output_n])
        #todo inverse the input
        self.input_chebyshev_coef = -input_chebyshev_coef
        self.output_chebyshev_coef = output_chebyshev_coef


    def __len__(self):
        return np.shape(self.input_chebyshev_coef)[0]
        # return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_chebyshev_coef[item], self.output_chebyshev_coef[item], self.all_seqs[item]
        # return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]

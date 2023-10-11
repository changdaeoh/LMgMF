'''
    borrowed from https://github.com/khanhnamle1994/MetaRec
'''

import torch
from sklearn.utils import shuffle
import numpy as np
import scipy.sparse as sp

from collections import defaultdict
import warnings
import os.path as osp 
warnings.filterwarnings('ignore')


# Initialize a Loader class
class Loader():
    # Set the iterator
    current = 0

    def __init__(self, x, y, batchsize=512, do_shuffle=True, out_np=False):
        """
        :param x: features
        :param y: target
        :param batchsize: batch size = 1024
        :param do_shuffle: shuffle mode turned on
        """
        self.shuffle = shuffle
        self.x = x
        self.y = y
        self.batchsize = batchsize
        self.out_np = out_np

        if len(self.y) > batchsize:
            self.batches = range(0, len(self.y), batchsize)
        else:
            self.batches = [0]

        if do_shuffle:
            # Every epoch re-shuffle the dataset
            self.x, self.y = shuffle(self.x, self.y)

    def __iter__(self):
        # Reset & return a new iterator
        self.x, self.y = shuffle(self.x, self.y, random_state=0)
        self.current = 0
        return self

    def __len__(self):
        # Return the number of batches
        if len(self.x) > self.batchsize:
            return int(len(self.x) / self.batchsize)
        else:
            1

    def __next__(self):
        # Update iterator and stop iteration until the batch size is out of range
        n = self.batchsize if len(self.y) > self.batchsize else len(self.y)
        if self.current != 0:
            if self.current + n >= len(self.y):
                raise StopIteration
        i = self.current
        self.current += n
        
        if self.out_np:
            xs = self.x[i:i + n]
            ys = self.y[i:i + n]
        else:
            xs = torch.from_numpy(self.x[i:i + n])
            ys = torch.from_numpy(self.y[i:i + n])
        return xs, ys


#! non-rating, existence-based recsys
n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)

def statistics(train_data, valid_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    #if args.dataset not in ['yelp2018']:
    #n_items -= n_users
    # remap [n_users, n_users+n_items] to [0, n_items]
    # train_data[:, 1] -= n_users
    # valid_data[:, 1] -= n_users
    # test_data[:, 1] -= n_users

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))


def build_sparse_graph(data_cf):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))
    return _bi_norm_lap(mat)


def load_data(model_args):
    global args
    args = model_args

    read_cf = lambda fn: np.load(fn).astype(np.int32)

    print('reading train and test user-item set ...')
    train_cf = read_cf(osp.join(args.data_root, f'rating_sub_train_seed{args.seed}.npy'))
    test_cf = read_cf(osp.join(args.data_root, f'rating_sub_test_seed{args.seed}.npy'))
    valid_cf = test_cf

    statistics(train_cf, valid_cf, test_cf)

    print('building the adj mat ...')
    norm_mat = build_sparse_graph(train_cf)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
    }
    user_dict = {
        'train_user_set': train_user_set,
        'valid_user_set': None,
        'test_user_set': test_user_set,
    }

    print('loading over ...')
    return train_cf, user_dict, n_params, norm_mat
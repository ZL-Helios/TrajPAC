import numpy as np


# ======================================================================
# ------------------------- HELPER FUNCTIONS ---------------------------
# ======================================================================


def dset2scene(dataset):
    if dataset == 'eth':
        test = ['biwi_eth']
    elif dataset == 'hotel':
        test = ['biwi_hotel']
    elif dataset == 'zara1':
        test = ['crowds_zara01']
    elif dataset == 'zara2':
        test = ['crowds_zara02']
    elif dataset == 'univ':
        test = ['students001', 'students003']
    return test[-1]



def get_mask(pid, pids, attack='basic'):
    mask = get_one_hot(idx=pids.index(pid), size=len(pids))
    if attack == 'basic':
        pass
    elif attack == 'env':
        mask = ~mask
    elif attack == 'full':
        mask[:] = True
    return mask


def get_one_hot(idx, size):
    onehot = np.zeros(size, dtype=bool)
    onehot[idx] = True
    return onehot


def expand_dims(x, n):
    num_expansions = n-x.ndim
    assert num_expansions > 0
    for i in range(num_expansions):
        x = np.expand_dims(x, axis=-1)
    return x
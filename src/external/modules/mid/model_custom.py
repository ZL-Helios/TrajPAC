import torch
import numpy as np
import yaml

from easydict import EasyDict

# MID modules
from mid import MID

# Custom modules
from modules.mid.dataset_custom import Dataset_Custom


class MID_Custom(object):
    
    def __init__(self, model, config) -> None:
        super().__init__()
        self.model = model      # not MID. It should be Autoencoder
        self.config = config


    @classmethod
    def from_args(cls, args):
        with open(args['MID'].config) as f:
            config = yaml.safe_load(f)
        config['dataset'] = args['DeepPAC'].dataset
        config['exp_name'] = args['MID'].config.split('/')[-1].split('.')[0]
        config['eval_at'] = args['MID'].checkpoint
        config['eval_mode'] = args['MID'].eval
        config = EasyDict(config)
        scenario = MID(config)

        # set some environment variables based on the scenario
        mid = cls(scenario.model, config)
        mid.hyperparams = scenario.hyperparams
        mid.eval_env = scenario.eval_env
        mid.eval_dataset = scenario.eval_dataset
        return mid


    def predict(self, batch, future, nodeType='PEDESTRIAN', pred_horizon=12, num_samples=20, bestof=True, st_inplace=False, verbose=False):
        """
        Returns the best-of-K predicted trajectory given by MID
        batch:      input batch data
        future:     (B, 12, 2) numpy array of future trajectories
        st_inplace: whether to standardize the future trajectory in-place or not
        """
        with torch.no_grad():
            # get predicted traj at each (ts, node) pair
            pred = self.model.generate(batch,   # (K, B, 12, 2)
                                       node_type=nodeType,
                                       num_points=pred_horizon,
                                       sample=num_samples,
                                       bestof=bestof)

        displacement = np.sqrt(np.sum((pred - future)**2, axis=-1))     # (K, B, 12)
        displacement = np.where(np.isnan(displacement), 10*np.ones_like(displacement), displacement)

        ade = np.mean(displacement, axis=-1)        # (K, B)
        fde = displacement[:,:,-1]                  # (K, B)

        idx_best_ade = np.argmin(ade, axis=0)       # (B,)
        idx_best_fde = np.argmin(fde, axis=0)       # (B,)

        pred_best_ade = pred[idx_best_ade, np.arange(len(idx_best_ade))]        # (B, 12, 2)
        pred_best_fde = pred[idx_best_fde, np.arange((len(idx_best_fde)))]      # (B, 12, 2)

        if verbose:
            best_ade = ade[idx_best_ade, np.arange(len(idx_best_ade))]  # (B,)
            best_ade = np.mean(best_ade)/0.6 if self.config.dataset == 'eth' else np.mean(best_ade)
            print('Average discplacement error best: {:0.3f}'.format(best_ade))

        scale = 0.6 if self.config.dataset == 'eth' else 1
        if st_inplace:  # in-place transform of future
            future /= scale
        return pred_best_ade / scale
        # rescale predictions (data was originally scaled in process_data.py)


    def forcast(self, args, x=None, **kwargs):

        # get the data dict.
        dataset = Dataset_Custom(self.eval_env,
                                 self.eval_dataset,
                                 self.hyperparams,
                                 self.config)
        idx = args['DeepPAC'].index
        data = dataset[idx]

        # prepare inputs
        if x is not None:
            batch = combine(data['past_df'], data['neighbors_df'])[0]   # (B, 8, 6)
            batch[:,:,:2] = np.copy(x)[:]                               # (B, 8, 6)
            mask = get_mask(len(batch), args['DeepPAC'].attack_scope)   # (B,) bool mask
            past_df = batch[:1][mask[:1]]                               # (1, 8, 6)
            nb_df = batch[1:][mask[1:]]                                 # (N, 8, 6)
        else:
            past_df = None; nb_df = None

        future = np.copy(data['future'])    # (1, 12, 2)
        batch_st = dataset.standardize_batch(data, st_agent=past_df, st_neigh=nb_df)

        # make prediction
        pred = self.predict(batch_st,      # (1, 12, 2)
                            future,
                            nodeType='PEDESTRIAN',
                            pred_horizon=12,
                            num_samples=20,
                            bestof=True,
                            st_inplace=False,
                            **kwargs)

        # for testing only. can remove later...
        if x is not None:
            batch_raw = combine(data['past_df'], data['neighbors_df'])[0]
            assert np.all(abs(batch-batch_raw)[:,:,:2]<=args['DeepPAC'].radius+1e-4)
            assert np.all(abs(batch-batch_raw)[:,:,2:]==0)
        return pred


    def attack(self, args, type='pgd', **kwargs):
        raise NotImplementedError




# ======================================================================
# ------------------------- HELPER FUNCTIONS ---------------------------
# ======================================================================

def combine(listx, listy):
    """
    Combines two lists of numpy arrays into one list of the concatenated numpy array.
    The arrays are concatenated along the first index of the larger array
    """
    assert len(listx)==len(listy)
    xy = []
    for i in range(len(listx)):
        x = listx[i]    # (8, 6)
        y = listy[i]    # (N, 8, 6)
        if len(x.shape) > len(y.shape):
            y = np.broadcast_to(y, x.shape[1:])[np.newaxis,...]   # (1, x.shape[1:])
        elif len(y.shape) > len(x.shape):
            x = np.broadcast_to(x, y.shape[1:])[np.newaxis,...]   # (1, y.shape[1:])
        else:
            raise NotImplementedError
        z = np.concatenate((x, y), axis=0)  # (Ni+1, 8, 6)
        xy.append(z)

    return xy   # (i=B, Ni+1, 8, 6)


def get_mask(bsize, attack='basic'):
    mask = np.full(bsize, True)
    if attack == 'basic':
        mask[1:] = False
    elif attack == 'env':
        mask[0] = False
    elif attack == 'full':
        pass
    return mask

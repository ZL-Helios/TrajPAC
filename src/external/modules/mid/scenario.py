# NOTE: The current working directory when executing this module should be in the 
#       MID project workspace. It is not in the DeepPAC workspace!

import math
import torch
import numpy as np
import yaml

from easydict import EasyDict
from tqdm import tqdm

# MID modules
from mid import MID

# Custom modules
from modules.mid.model_custom import MID_Custom
from modules.mid.dataset_custom import Dataset_Custom

CPU = torch.device('cpu')
GPU = torch.device('cuda')


class Scenario(MID):
    """
    Class that acts like a dataloader for generating scenario data (i.e., noisy data about
    the center of L-inf B(x̂,r)). The dataset is generated based on MID, and the
    center point x̂ can be selected by indexing. 
    """

    def __init__(self, args) -> None:

        # define Scenario arguments
        self.args = args
        self.device = GPU if args['DeepPAC'].gpu else CPU
        self.bsize = args['DeepPAC'].bsize
        self.attack_scope = args['DeepPAC'].attack_scope
        DATASET = args['DeepPAC'].dataset

        # define DeepPAC parameters
        self.radius = args['DeepPAC'].radius
        self.error = args['DeepPAC'].epsilon
        self.significance = args['DeepPAC'].eta
        self.FThreshold = args['DeepPAC'].FThreshold
        self.SThreshold = args['DeepPAC'].SThreshold
        self.TThreshold = math.ceil(2/self.error*(math.log(1/self.significance)+1))

        # define MID parameters
        self.config = self.load_MID_config()
    
        # build the model and datasets (based on eval data)
        super(Scenario, self).__init__(self.config)
        self.MID = MID_Custom(self.model, self.config)
        self.dataset = Dataset_Custom(self.eval_env, self.eval_dataset, self.hyperparams, self.config)

        # NOTE zara1
        # num parameters: 9651428


    def load_MID_config(self):
        with open(self.args['MID'].config) as f:
            config = yaml.safe_load(f)
        config['dataset'] = self.args['DeepPAC'].dataset
        config['exp_name'] = self.args['MID'].config.split('/')[-1].split('.')[0]
        config['eval_at'] = self.args['MID'].checkpoint
        config['eval_mode'] = self.args['MID'].eval
        config = EasyDict(config)
        return config



    def __getitem__(self, idx):
        print(f'generating noisy data around sample {idx}...')
        
        # Get the batch data containing the given idx=(fid,pid) pair
        data = self.dataset[idx]
        batch = combine(data['past_df'], data['neighbors_df'])[0]   # (B, 8, 6)
        mask = self.get_mask(len(batch), self.attack_scope)         # (B,) bool mask

        # Get the center point x̂ of the L-inf ball B(x̂,r) for a given (max) batchsize
        past = np.copy(data['past'])        # (1, 8, 2) np array
        future = np.copy(data['future'])    # (1, 12, 2) np array
        pred = self.predict(data,           # (1, 12, 2) np array
                            future,
                            st_wrt=None,
                            st_nb_wrt=None,
                            num_samples=20,
                            inplace=True,   # NOTE: in-place transform of future
                            verbose=True)
        print('\tBatch: ', batch.shape)
        print('\tPast:  ', past.shape)
        print('\tFuture:', future.shape)
        #assert np.max(abs(future-data['future']))>0


        # Generate scenario optimization samples
        scenarios = self.create_scenarios(batch, self.radius)       # (T1+T2+T3, B, 8, 6)
        phases = self.split_into_phases(scenarios)                  # [(Ti, B, 8, 6)]


        # Generate scenario/noisy outputs
        print('preparing focused data...')
        future_cp = np.copy(data['future'])
        pred_phase = []
        for phase in phases:                    # (T, B, 8, 6)
            pred_batch = []
            for batch_ in tqdm(phase):          # (B, 8, 6)
                past_df = batch_[:1][mask[:1]]  # (1, 8, 6)
                nb_df = batch_[1:][mask[1:]]    # (N, 8, 6)
                recon = self.predict(data,
                                     future_cp,
                                     st_wrt=past_df,
                                     st_nb_wrt=nb_df,
                                     num_samples=20,
                                     inplace=False,
                                     verbose=False)
                pred_batch.append(recon)            # (1, 12, 2)
            pred_phase.append(np.stack(pred_batch)) # (T, 1, 12, 2)

        assert np.max(abs(future_cp-data['future']))==0
        assert np.array_equal(batch, combine(data['past_df'], data['neighbors_df'])[0])


        # Compile the data (use equilareal tranformed data for future and pred)
        center, focused = {}, [{}, {}, {}]
        past_phase = self.split_into_phases(scenarios[:,mask,:,:2])
        batch = batch[mask,:,:2]

        center['past'] = batch          # (B, 8, 2)
        center['future'] = future       # (1, 12, 2)
        center['pred'] = pred           # (1, 12, 2)
        for i in range(3):
            focused[i]['past'] = past_phase[i]  # (T, B, 8, 2)
            focused[i]['pred'] = pred_phase[i]  # (T, 1, 12, 2)

        return {
            'center': center,
            'noisy': focused,
            'fids': data['fids'][mask],   # (B, 8)
            'pids': data['pids'][mask],   # (B,)
        }



    def predict(self, data, future, st_wrt=None, st_nb_wrt=None, num_samples=20, inplace=False, verbose=False):
        """
        Makes best-of-K predictions based on the past trajectory and future
        data:       data dictionary
        future:     (1, 12, 2)  future traj
        st_wrt:     (1, 8, 6)   past traj to standardize to
        st_nb_wrt:  (N, 8, 6)   neighbor traj to standardize to
        """
        # set empty arrays to None
        st_wrt = None if (st_wrt is not None and len(st_wrt) == 0) else st_wrt
        st_nb_wrt = None if (st_nb_wrt is not None and len(st_nb_wrt) == 0) else st_nb_wrt
        # batch the neighbor nodes for input into get_batch
        if st_nb_wrt is not None: st_nb_wrt = [st_nb_wrt]   # (1, N, 8, 6)

        if st_wrt is None and st_nb_wrt is None:
            batch = data['batch']
        else:
            batch = self.dataset.get_batch(scene=data['scene'],
                                           timesteps=data['timesteps'],
                                           nodes=data['nodes'],
                                           scene_graphs=data['scene_graphs'],
                                           wrt=st_wrt,
                                           nb_wrt=st_nb_wrt)
        pred = self.MID.predict(batch,
                                future,
                                nodeType='PEDESTRIAN',
                                pred_horizon=12,
                                num_samples=num_samples,
                                bestof=True,
                                st_inplace=inplace,
                                verbose=verbose)
        return pred


    def predict_adversarial(self, idx, batch_adv, num_predictions=1, **kwargs):
        """
        Gets the prediction for an adversarial sample at idx
        batch_adv: (B, 8, 2) ndarray adversarial sample
        """
        print(f'generating adversarial prediction for sample {idx}...')

        # get the batch data containing the given idx=(fid,pid) pair
        data = self.dataset[idx]
        batch = combine(data['past_df'], data['neighbors_df'])[0]   # (B, 8, 6)
        batch[:,:,:2] = np.copy(batch_adv)[:]                       # (B, 8, 6)
        mask = self.get_mask(len(batch), self.attack_scope)         # (B,) bool mask

        # make prediction
        future = np.copy(data['future'])    # (1, 12, 2)
        past_df = batch[:1][mask[:1]]       # (1, 8, 6)
        nb_df = batch[1:][mask[1:]]         # (N, 8, 6)
        pred = []
        for i in tqdm(range(num_predictions)):
            recon = self.predict(data,      # (1, 12, 2)
                                future,
                                st_wrt=past_df,
                                st_nb_wrt=nb_df,
                                num_samples=20,
                                inplace=False,
                                **kwargs)
            pred.append(recon)
        pred = np.stack(pred)

        batch_raw = combine(data['past_df'], data['neighbors_df'])[0]
        assert np.all(abs(batch-batch_raw)[:,:,:2]<=self.radius+1e-4)
        assert np.all(abs(batch-batch_raw)[:,:,2:]==0)
        return pred


    def get_mask(self, bsize, attack='basic'):
        mask = np.full(bsize, True)
        if attack == 'basic':
            mask[1:] = False
        elif attack == 'env':
            mask[0] = False
        elif attack == 'full':
            pass
        return mask

    def create_scenarios(self, x, radius):
        """
        Function to generate n iid uniform samples within B(x̂,r) for scenario optimization
        x:              (B, *...) center of the L-inf ball (B, 8, 6)
        radius:         radius of the L-inf ball B(x̂,r)
        num_scenarios:  number of scenarios/samples
        """
        num_scenarios = self.FThreshold + self.SThreshold + self.TThreshold
        radius = radius*np.ones_like(x)
        radius[:,:,2:] = 0  # only add noise to positions
        radius[x==0.0] = 0  # ignore padded positions
        x_l = x - radius
        x_u = x + radius
        return (x_u-x_l)*np.random.rand(num_scenarios, *x.shape) + x_l

    def split_into_phases(self, x):
        """ Function that splits (T1+T2+T3, B, 8, 2) into a list of data for 3 phases """
        y = x.copy()
        del x
        return [
            y[:self.FThreshold],
            y[self.FThreshold:self.FThreshold + self.SThreshold],
            y[self.FThreshold + self.SThreshold:]
        ]



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

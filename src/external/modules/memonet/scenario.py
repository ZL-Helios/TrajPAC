# NOTE: The current working directory when executing this module should be in the 
#       MemoNet project workspace. It is not in the DeepPAC workspace!

import os
import math
import numpy as np
import torch
from tqdm import tqdm

# Memonet modules
from utils.config import Config

# Custom modules
from modules.memonet.model_custom import MemoNet_Custom
from modules.memonet.dataset_custom import Dataset_Custom

CPU = torch.device('cpu')
GPU = torch.device('cuda')


class Scenario():
    """
    Class that acts like a dataloader for generating scenario data (i.e., noisy data about
    the center of L-inf B(x̂,r)). The dataset used is from the ___ class, and the
    center point x̂ can be selected by indexing. 
    """

    def __init__(self, args):
        super(Scenario, self).__init__()

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

        # define MemoNet parameters
        self.cfg = Config(cfg_id=DATASET, cfg_info='', tmp=False, create_dirs=True)

        # load MemoNet model
        #self.memoNet = self._load_model(self.cfg)
        self.memoNet = MemoNet_Custom(self.cfg, self.device)
        self.memoNet.eval()

        # load dataset
        self.dataset = self._load_dataset(self.cfg)

        # NOTE zara1
        # num parameters: 5281252


    def _load_model(self, cfg):
        # load MemoNet model
        MODEL_PATH = cfg.model_encdec
        memoNet = MemoNet_Custom(cfg)
        memoNet.model_encdec.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        memoNet.to(self.device)
        return memoNet

    def _load_dataset(self, cfg):
        log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')
        dataset = Dataset_Custom(cfg, log,
                                split = 'train' if self.args['DeepPAC'].train else 'test',
                                phase = 'training' if self.args['DeepPAC'].train else 'testing')
        return dataset



    def __getitem__(self, idx):
        print(f'generating noisy data around sample {idx}...')
        idx = (idx[0]//10, idx[1]) # MemoNet's eth-ucy fids are scaled by 10
        assert self.cfg.traj_scale == 1

        # Get the batch data containing the given idx=(fid,pid) pair
        data = self.dataset[idx]
        mask_noise = get_mask(data['valid_id'], idx[1], self.attack_scope)  # (B,)
        mask_pred = get_one_hot(data['valid_id'], idx[1])                   # (B,)

        # Get the center point x̂ of the L-inf ball B(x̂,r)
        past = torch.stack(data['pre_motion_3D']).to(self.device)       # (B, 8, 2)
        future = torch.stack(data['fut_motion_3D']).to(self.device)     # (B, 12, 2)
        pred = self.memoNet.predict(past,                               # (1, 12, 2) masked prediction
                                    future,
                                    inplace=True,
                                    verbose=True)[mask_pred].cpu().numpy()
        print('\tPast:  ', past.cpu().numpy().shape)
        print('\tFuture:', future.cpu().numpy().shape)
        assert torch.max(abs(future-torch.stack(data['fut_motion_3D']).to(self.device)))>0


        # Generate scenario optimization samples and split into focused learning phases
        scenarios = self.create_scenarios(past, self.radius, mask_noise)    # (T1+T2+T3, B, 8, 2)
        phases = self.split_into_phases(scenarios)                          # [(T1, B, 8, 2),...]
        assert torch.all((past-scenarios[-1])[mask_noise]!=0)
        assert torch.all((past-scenarios[-1])[~mask_noise]==0)


        # Generate scenario/noisy outputs
        print('preparing focused data...')
        future_cp = torch.stack(data['fut_motion_3D']).to(self.device)
        pred_phase = []
        for phase in phases:            # (T, B, 8, 2)
            pred_batch = []
            for batch in tqdm(phase):   # (B, 8, 2)
                recon = self.memoNet.predict(batch,
                                             future_cp,
                                             inplace=False,
                                             verbose=False)[mask_pred].cpu().numpy()
                pred_batch.append(recon)                # (1, 12, 2)
            pred_phase.append(np.stack(pred_batch))     # (T, 1, 12, 2)
        assert torch.equal(future_cp, torch.stack(data['fut_motion_3D']).to(self.device))


        # Compile the data (use equilareal tranformed data for future and pred)
        center, focused = {}, [{}, {}, {}]
        center['past'] = past[mask_noise].cpu().numpy()         # (B, 8, 2)
        center['future'] = future[mask_pred].cpu().numpy()      # (1, 12, 2)
        center['pred'] = pred                                   # (1, 12, 2)
        for i in range(3):
            focused[i]['past'] = phases[i][:,mask_noise].cpu().numpy()  # (T, B, 8, 2)
            focused[i]['pred'] = pred_phase[i]                          # (T, 1, 12, 2)

        return {
            'center': center,
            'noisy': focused,
            'fids': data['fids'][mask_noise],   # (B, 8)
            'pids': data['pids'][mask_noise],   # (B,)
        }


    def predict_adversarial(self, idx, past_adv, num_predictions=1, **kwargs):
        """
        Gets the prediction for an adversarial sample at idx
        past: (B, 8, 2) ndarray adversarial sample
        """
        print(f'generating adversarial prediction for sample {idx}...')
        idx = (idx[0]//10, idx[1]) # MemoNet's eth-ucy fids are scaled by 10
        assert self.cfg.traj_scale == 1

        # get the batch data containing the given idx=(fid,pid) pair
        data = self.dataset[idx]
        mask_noise = get_mask(data['valid_id'], idx[1], self.attack_scope)  # (B,)
        mask_agent = get_mask(data['valid_id'], idx[1], 'basic')            # (B,)

        # make prediction
        past = torch.from_numpy(past_adv).type(torch.float).to(self.device) # (B, 8, 2)
        future = torch.stack(data['fut_motion_3D']).to(self.device)         # (B, 12, 2)
        pred = []
        for i in tqdm(range(num_predictions)):
            recon = self.memoNet.predict(past,  # (1, 12, 2)
                                        future,
                                        inplace=False,
                                        **kwargs).cpu().numpy()[mask_agent]
            pred.append(recon)
        pred = np.stack(pred)
        assert torch.all(abs(past-torch.stack(data['pre_motion_3D']).to(self.device))<=self.radius+1e-4)
        assert torch.equal(future, torch.stack(data['fut_motion_3D']).to(self.device))
        return pred



    def create_scenarios(self, x, radius, mask=None):
        """
        Function to generate n iid uniform samples within B(x̂,r) for scenario optimization
        x:          (B, *...) center of the L-inf ball
        radius:     radius of the L-inf ball B(x̂,r)
        mask:       (optional) only add noise according to the given mask [False, True, True, ...]
        """
        num_scenarios = self.FThreshold + self.SThreshold + self.TThreshold
        if mask is not None:
            radius = expand_dims(radius*mask, n=x.dim())  # (B, 1...)        
        x_l = x - torch.from_numpy(radius).type(x.dtype).to(x.device)
        x_u = x + torch.from_numpy(radius).type(x.dtype).to(x.device)
        return ((x_u-x_l)*torch.rand(
            (num_scenarios, *x.shape),
            dtype = x.dtype,
            layout = x.layout,
            device = x.device) + x_l)


    def split_into_phases(self, x):
        """ Function that splits (T1+T2+T3, B, 8, 2) into a list of data for 3 phases """
        y = x.clone() if type(x)==torch.Tensor else x.copy()
        del x   # delete reference to original data
        return [
            y[:self.FThreshold],
            y[self.FThreshold:self.FThreshold + self.SThreshold],
            y[self.FThreshold + self.SThreshold:]
        ]


# ======================================================================
# ------------------------- HELPER FUNCTIONS ---------------------------
# ======================================================================

def get_mask(pids, pid, attack='basic'):
    # mask to determine which paths to add noise
    mask = get_one_hot(pids, pid)
    if attack=='basic':
        pass
    elif attack=='env':
        mask = ~mask
    elif attack=='full':
        mask[:] = True
    return mask

def get_one_hot(pids, pid):
    # one-hot mask for the predicted traj
    one_hot_idx = pids.index(pid)
    mask = np.zeros(len(pids), dtype=bool)
    mask[one_hot_idx] = True
    return mask

def expand_dims(x, n):
    num_expansions = n-x.ndim
    assert num_expansions > 0
    for i in range(num_expansions):
        x = np.expand_dims(x, axis=-1)
    return x
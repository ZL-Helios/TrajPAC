# NOTE: The current working directory when executing this module should be in the 
#       AgentFormer project workspace. It is not in the DeepPAC workspace!

import math
import torch
import numpy as np
from tqdm import tqdm

# AgentFormer modules
from utils.config import Config

# Custom modules
from modules.agentformer.model_custom import AgentFormer_Custom
from modules.agentformer.dataset_custom import Dataset_Custom

CPU = torch.device('cpu')
GPU = torch.device('cuda')


class Scenario:
    """
    Class that acts like a dataloader for generating scenario data (i.e., noisy data about
    the center of L-inf B(x̂,r)). The dataset is generated based on Trajectron++, and the
    center point x̂ can be selected by indexing. 
    """
    def __init__(self, args) -> None:
        super(Scenario, self).__init__()

        # define Scenario arguments
        self.args = args
        self.device = GPU if args['DeepPAC'].gpu else CPU
        self.bsize = args['DeepPAC'].bsize                  # NOTE unused
        self.attack_scope = args['DeepPAC'].attack_scope
        DATASET = args['DeepPAC'].dataset

        # define DeepPAC parameters
        self.radius = args['DeepPAC'].radius
        self.error = args['DeepPAC'].epsilon
        self.significance = args['DeepPAC'].eta
        self.FThreshold = args['DeepPAC'].FThreshold
        self.SThreshold = args['DeepPAC'].SThreshold
        self.TThreshold = math.ceil(2/self.error*(math.log(1/self.significance)+1))
        self.prediction_type = args['DeepPAC'].prediction_type

        # define AgentFormer and dataset
        self.cfg = Config(f'{DATASET}_agentformer')
        self.dataset = Dataset_Custom(self.cfg)
        self.agentformer = AgentFormer_Custom(self.cfg, self.device)
        self.agentformer.eval()

        # NOTE zara1
        # num parameters: 5972098 (agentformer) + 591872 (dlow) = 6563970



    def __getitem__(self, idx):
        print(f'generating noisy data around sample {idx}...')
        idx = (idx[0]//10, idx[1]) # AgentFormer's eth-ucy fids are scaled by 10

        # Get the batch data containing the given idx=(fid,pid) pair
        data = self.dataset[idx]
        scale = self.cfg.traj_scale     # NOTE: ETH scale == 2; ZARA scale == 1
        mask_noise = get_mask(idx[1], data['valid_id'], self.attack_scope)
        mask_agent = get_mask(idx[1], data['valid_id'], 'basic')

        # Get the center point x̂ of the L-inf ball B(x̂,r)
        past = torch.stack(data['pre_motion_3D']).to(self.device) * scale           # (B, 8, 2)
        future = torch.stack(data['fut_motion_3D']).to(self.device) * scale         # (B, 12, 2)
        pred = self.agentformer.predict(data,
                                        type=self.prediction_type,
                                        K=20,
                                        verbose=True).cpu().numpy()     # (B, 12, 2)
        print('\tPast:  ', past.cpu().numpy().shape)
        print('\tFuture:', future.cpu().numpy().shape)
        assert torch.all(future - self.agentformer.data['fut_motion_orig']*scale == 0)


        # Generate scenario optimization samples and split into focused learning phases
        scenarios = self.create_scenarios(past.cpu(), self.radius, mask_noise)  # (T1+T2+T3, B, 8, 2)
        phases = self.split_into_phases(scenarios)                              # [(T, B, 8, 2)...]
        assert torch.all((past.cpu() - scenarios[0])[~mask_noise] == 0)
        assert torch.all((past.cpu() - scenarios[0])[mask_noise] < self.radius)


        # Generate scenario/noisy outputs
        print('preparing focused data...')
        recon_phases = []
        for phase in phases:
            recon_phase = []
            for batch in tqdm(phase):
                data['pre_motion_3D'] = list(batch/scale)
                recon = self.agentformer.predict(data,
                                                 type=self.prediction_type,
                                                 K=20,
                                                 verbose=False).cpu().numpy()
                recon_phase.append(recon)                   # (B, 12, 2)
            recon_phases.append(np.stack(recon_phase))      # (T, B, 12, 2)
        assert torch.all(future.cpu() - torch.stack(data['fut_motion_3D'])*scale == 0)


        # Compile the data (use equilareal tranformed data for future and pred)
        center, focused = {}, [{}, {}, {}]
        center['past'] = past[mask_noise].cpu().numpy()                     # (B, 8, 2)
        center['future'] = future[mask_agent].cpu().numpy()                 # (1, 12, 2)
        center['pred'] = pred[mask_agent]                                   # (1, 12, 2)
        for i in range(3):
            focused[i]['past'] = phases[i][:,mask_noise].cpu().numpy()      # (T, B, 8, 2)
            focused[i]['pred'] = recon_phases[i][:,mask_agent]              # (T, 1, 12, 2)

        return {
            'center': center,
            'noisy': focused,
            'fids': data['fids'][mask_noise],   # (B, 8)
            'pids': data['pids'][mask_noise],   # (B,)
        }


    def predict_adversarial(self, idx, past, num_predictions=1, **kwargs):
        """
        Gets the prediction for an adversarial sample at idx
        past: (B, 8, 2) ndarray adversarial sample
        """
        print(f'generating adversarial prediction for sample {idx}...')
        idx = (idx[0]//10, idx[1]) # AgentFormer's eth-ucy fids are scaled by 10
        past = torch.from_numpy(past).type(torch.float)

        # get the batch data containing the given idx=(fid,pid) pair
        data = self.dataset[idx]
        scale = self.cfg.traj_scale
        mask_noise = get_mask(idx[1], data['valid_id'], self.attack_scope)
        mask_agent = get_mask(idx[1], data['valid_id'], 'basic')
        assert torch.all(abs(past-torch.stack(data['pre_motion_3D'])*scale) <= self.radius+1e-4)

        # make prediction
        data['pre_motion_3D'] = list(past/scale)        # (B, 8, 2)
        pred = []
        for i in tqdm(range(num_predictions)):
            recon = self.agentformer.predict(data,      # (1, 12, 2)
                                            K=20,
                                            **kwargs).cpu().numpy()[mask_agent]
            pred.append(recon)
        pred = np.stack(pred)
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
            radius = expand_dims(radius*mask, n=x.dim())
        x_l = x - radius
        x_u = x + radius
        return ((x_u-x_l)*torch.rand(
            (num_scenarios, *x.shape),
            dtype = x.dtype,
            layout = x.layout,
            device = x.device) + x_l).type(x.dtype)


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
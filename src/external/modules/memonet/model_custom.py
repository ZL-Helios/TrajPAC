import os
import math
import torch
import numpy as np

# Memonet modules
from utils.config import Config
from models.model_test_trajectory import MemoNet

# Custom modules
from modules.memonet.dataset_custom import Dataset_Custom

CPU = torch.device('cpu')
GPU = torch.device('cuda')


class MemoNet_Custom(MemoNet):
    """
    MemoNet inherited class that lets us make (best) trajectory predictions
    """
    def __init__(self, cfg, device=CPU):
        super(MemoNet_Custom, self).__init__(cfg)
        self.cfg = cfg
        self.device = device
        # NOTE: added --load MemoNet model
        MODEL_PATH = cfg.model_encdec
        self.model_encdec.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        self.to(device)
        

    @classmethod
    def from_args(cls, args):
        DATASET = args['DeepPAC'].dataset
        device = GPU if args['DeepPAC'].gpu else CPU
        cfg = Config(cfg_id=DATASET, cfg_info='', tmp=False, create_dirs=True)
        return cls(cfg, device)


    def num_parameters(self, only_trainable=False):
        return sum(p.numel() for p in self.parameters() if ((not only_trainable) or p.requires_grad))


    def rotate_traj(self, past, future, past_abs):
        past_diff = past[:, 0]
        past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0]+1e-5))
        past_theta = torch.where((past_diff[:, 0]<0), past_theta+math.pi, past_theta)

        rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
        rotate_matrix[:, 0, 0] = torch.cos(past_theta)
        rotate_matrix[:, 0, 1] = torch.sin(past_theta)
        rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
        rotate_matrix[:, 1, 1] = torch.cos(past_theta)

        past_after = torch.matmul(rotate_matrix, past.transpose(1, 2)).transpose(1, 2)
        future_after = torch.matmul(rotate_matrix, future.transpose(1, 2)).transpose(1, 2)
        
        b1 = past_abs.size(0)
        b2 = past_abs.size(1)
        for i in range(b1):
            past_diff = (past_abs[i, 0, 0]-past_abs[i, 0, -1]).unsqueeze(0).repeat(b2, 1)
            past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0]+1e-5))
            past_theta = torch.where((past_diff[:, 0]<0), past_theta+math.pi, past_theta)

            rotate_matrix = torch.zeros((b2, 2, 2)).to(past_theta.device)
            rotate_matrix[:, 0, 0] = torch.cos(past_theta)
            rotate_matrix[:, 0, 1] = torch.sin(past_theta)
            rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
            rotate_matrix[:, 1, 1] = torch.cos(past_theta)
            # print(past_abs.size())
            past_abs[i] = torch.matmul(rotate_matrix, past_abs[i].transpose(1, 2)).transpose(1, 2)
        # print('-'*50)
        # print(past_abs.size())
        return past_after, future_after, past_abs


    def predict(self, past, future, inplace=False, verbose=False):
        """
        Function that returns the best predicted trajectories based on the past trajectory
        past:   (N, 8, 2)   past trajectories (8 frames) for N people
        future: (N, 12, 2)  future trajectories (12 frames) for N people
        """
        with torch.no_grad():

            # normalize the traj relative to current position
            last_frame = past[:, -1:]                   # (N, 1, 2)
            past_normalized = past - last_frame
            fut_normalized = future - last_frame

            past_abs = past.unsqueeze(0).repeat(past.size(0), 1, 1, 1)  # (N, N, 8, 2)
            past_centroid = past[:, -1:, :].unsqueeze(1)                # (N, 1, 8, 2)
            past_abs = past_abs - past_centroid                         # (N, N, 8, 2)

            # scale the past trajectory data
            scale = 1
            if self.cfg.scale.use:
                scale = torch.mean(torch.norm(past_normalized[:, 0], dim=1)) / 3
                if scale<self.cfg.scale.threshold:
                    scale = 1
                else:
                    if self.cfg.scale.type == 'divide':
                        scale = scale / self.cfg.scale.large
                    elif self.cfg.scale.type == 'minus':
                        scale = scale - self.cfg.scale.large
                if self.cfg.scale.type=='constant':
                    scale = self.cfg.scale.value
                past_normalized = past_normalized / scale
                past_abs = past_abs / scale

            # rotate the trajectory data
            if self.cfg.rotation:
                past_normalized, fut_normalized, past_abs = self.rotate_traj(past_normalized, fut_normalized, past_abs)
            end_pose = past_abs[:, :, -1]     # (N, N, 2)


            # make MemoNet prediction (K=20 traj/cluster predictions)
            prediction = self.forward(past_normalized, past_abs, end_pose)  # (N, 20, 12, 2)
            prediction = prediction.data * scale

            # compute l2 distance errors
            future_rep = fut_normalized.unsqueeze(1).repeat(1, 20, 1, 1)    # (N, 20, 12, 2)
            distances = torch.norm(prediction - future_rep, dim=-1)         # (N, 20, 12)
            distances = torch.where(torch.isnan(distances), torch.full_like(distances, 10), distances)

            fde = distances[:,:,-1]                     # (N, 20) FDE
            ade = torch.mean(distances, dim=-1)         # (N, 20) ADE

            idx_best_fde = torch.argmin(fde, dim=-1)    # (N,)
            idx_best_ade = torch.argmin(ade, dim=-1)    # (N,)

            pred_fde = prediction[torch.arange(len(idx_best_fde)), idx_best_fde]   # (N, 12, 2)
            pred_ade = prediction[torch.arange(len(idx_best_ade)), idx_best_ade]   # (N, 12, 2)

            if verbose:
                best_ade = torch.sum((pred_ade - fut_normalized)**2, dim=-1).sqrt_().mean()
                print('Average discplacement error best: {:0.3f}'.format(best_ade))

        if inplace:
            future[:] = fut_normalized  # apply in-place normalization on future traj
        return pred_ade



    def forcast(self, args, x=None, **kwargs):
        """
        black box forward propagation of MemoNet model
        idx:    (fid, pid) tuple of the input trajectory
        x:      (B, 8, 2) ndarray of all past trajectories at the given frame (optional)
        """

        idx = args['DeepPAC'].index
        idx = (idx[0]//10, idx[1])      # MemoNet's eth-ucy fids are scaled by 10
        assert self.cfg.traj_scale == 1

        # get the batch data
        log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
        data = Dataset_Custom(self.cfg, log, split='test', phase='testing')[idx]
        mask_agent = get_mask(data['valid_id'], idx[1], 'basic')            # (B,)

        # prepare inputs
        future = torch.stack(data['fut_motion_3D']).to(self.device)         # (B, 12, 2)
        if x is not None:
            past = torch.from_numpy(x).type(torch.float).to(self.device)    # (B, 8, 2)
        else:
            past = torch.stack(data['pre_motion_3D']).to(self.device)       # (B, 8, 2)

        # make prediction
        self.eval()
        recon = self.predict(past,  # (1, 12, 2)
                            future,
                            inplace=False,
                            **kwargs).cpu().numpy()[mask_agent]

        assert torch.all(abs(past-torch.stack(data['pre_motion_3D']).to(self.device))<=args['DeepPAC'].radius+1e-4)
        assert torch.equal(future, torch.stack(data['fut_motion_3D']).to(self.device))
        return recon


    def attack(self, args, type='pgd', **kwargs):
        raise NotImplementedError



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


def _load_dataset(self, cfg):
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')
    dataset = Dataset_Custom(cfg, log,
                            split = 'train' if self.args['DeepPAC'].train else 'test',
                            phase = 'training' if self.args['DeepPAC'].train else 'testing')
    return dataset

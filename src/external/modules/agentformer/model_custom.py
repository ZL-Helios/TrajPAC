import random
import torch
import numpy as np
from tqdm import tqdm

# AgentFormer modules
from utils.config import Config
from model.model_lib import model_dict  # ad hoc solution to prevent circular dependency
from model.dlow import DLow

# Custom models
from modules.agentformer.dataset_custom import Dataset_Custom

CPU = torch.device('cpu')
GPU = torch.device('cuda')


class AgentFormer_Custom(DLow):

    def __init__(self, cfg: Config, device) -> None:
        super(AgentFormer_Custom, self).__init__(cfg)
        self.device = device
        self.set_device(device)
        if (epoch:=cfg.get_last_epoch()) > 0:
            cp_path = cfg.model_path % epoch
            print(f'loading model from checkpoint: {cp_path}')
            model_cp = torch.load(cp_path, map_location='cpu')
            self.load_state_dict(model_cp['model_dict'], strict=False)
        self.scale = cfg.traj_scale

    @classmethod
    def from_config(cls, cfg: Config, device=CPU):
        return cls(cfg, device)
    
    @classmethod
    def from_args(cls, args):
        DATASET = args['DeepPAC'].dataset
        device = GPU if args['DeepPAC'].gpu else CPU    # NOTE: CHECK IF WORKS??
        cfg = Config(f'{DATASET}_agentformer')
        return cls(cfg, device)

    def num_parameters(self, only_trainable=False):
        return sum(p.numel() for p in self.parameters() if ((not only_trainable) or p.requires_grad))


    def inference(self, mode, type='best_of', need_weights=False):
        """
        Overloaded inference function that changes self.main(mean=True) into
        self.main(mean=False) to allow generative predictions (not just from z=mean)
        """
        if type=='best_of':
            self.main(mean=False, need_weights=need_weights)
        elif type=='most_likely':
            self.main(mean=True, need_weights=need_weights)
        res = self.data[f'infer_dec_motion']    # (B, 20, 12, 2) for both 'recon' and 'infer'
        if mode == 'recon':
            res = res[:, 0]     # just select first of the K samples for reconstruction
        return res, self.data



    def predict(self, data, future=None, type='best_of', K=20, inplace=False, verbose=False):
        """
        Function that predicts the best-of-K future trajectories
        data:   data dictionary
        future: (optional) tensor of size (B, 12, 2) corresponding to the raw future traj
        """
        future = torch.stack(data['fut_motion_3D']).to(self.device) * self.scale if future is None else future
        future = torch.unsqueeze(future, dim=1)     # (B, 1, 12, 2)

        with torch.no_grad():
            self.set_data(data)     # transforms and sets the data
            pred = self.inference(mode='infer',
                                  type=type,
                                  need_weights=False)[0] * self.scale   # (B, 20, 12, 2)

            l2_dist = torch.sum((pred - future)**2, dim=-1).sqrt()     # (B, 20, 12)
            l2_dist = torch.where(torch.isnan(l2_dist), torch.full_like(l2_dist, 10), l2_dist)

            ade = torch.mean(l2_dist, dim=-1)   # (B, 20)
            fde = l2_dist[:,:,-1]               # (B, 20)

            idx_best_ade = torch.argmin(ade, dim=-1)    # (B,)
            idx_best_fde = torch.argmin(fde, dim=-1)    # (B,)

            pred_ade = pred[torch.arange(len(idx_best_ade)), idx_best_ade]  # (B, 12, 2)
            pred_fde = pred[torch.arange(len(idx_best_fde)), idx_best_fde]  # (B, 12, 2)

            if verbose:
                best_ade = ade[torch.arange(len(idx_best_ade)), idx_best_ade].mean()   # (B,)
                print('Average discplacement error best: {:0.3f}'.format(best_ade))

        if inplace:
            pass
        return pred_ade


    def forcast(self, args, x=None, **kwargs):
        """
        black box forward propagation of AgentFormer model
        idx:    (fid, pid) tuple of the input trajectory
        x:      (B, 8, 2) ndarray of all past trajectories at the given frame (optional)
        """
        idx = args['DeepPAC'].index
        idx = (idx[0]//10, idx[1])  # AgentFormer's eth-ucy fids are scaled by 10
        pred_type = args['DeepPAC'].prediction_type

        # get the data dict.
        data = Dataset_Custom(self.cfg)[idx]
        scale = self.cfg.traj_scale
        mask_agent = get_mask(idx[1], data['valid_id'], 'basic')

        # prepare inputs
        if x is not None:
            past = torch.from_numpy(x).type(torch.float)
            data['pre_motion_3D'] = list(past/scale)    # (B, 8, 2)
            #assert torch.all(abs(past-torch.stack(data['pre_motion_3D'])*scale) <= args['DeepPAC'].radius+1e-4)

        # make prediction
        self.eval()
        return self.predict(data,                       # (1, 12, 2)
                            type=pred_type,
                            K=20,
                            **kwargs).cpu().numpy()[mask_agent]



    def attack(self, args, type='pgd', **kwargs):

        idx = args['DeepPAC'].index
        idx = (idx[0]//10, idx[1])  # AgentFormer's eth-ucy fids are scaled by 10
        radius = args['DeepPAC'].radius
        self.pred_type = args['DeepPAC'].prediction_type

        # get the data dict.
        data = Dataset_Custom(self.cfg)[idx]
        scale = self.cfg.traj_scale
        #mask_agent = get_mask(idx[1], data['valid_id'], 'basic')

        adversary = self.pgd(data,
                             n_steps=30,
                             step_size=(radius/scale)/20,
                             step_norm='inf',
                             eps=radius/scale,
                             eps_norm='inf')
        return scale * np.array([x.detach().cpu().numpy() for x in adversary])



    def pgd(self, data, n_steps, step_size, step_norm, eps, eps_norm):
        """
        Computes projected gradient descent on the data to find adversarial examples.
        n_steps:    number of steps in gradient descent
        step_size:  size of each gradient step
        step_norm:  the norm/magnitude of the gradient step
        eps:        the attack radius
        eps_norm:   the norm of the ball of attack
        * Note that for AgentFormer the data['pre_motion_3D'] is scaled by /= scale !
          This means that our eps and step_size must account for this scale.
        """

        pre_motion_3D = data['pre_motion_3D']
        pre_motion_3D_original = []
        for i, past in enumerate(pre_motion_3D):
            pre_motion_3D_original.append(pre_motion_3D[i].detach().clone())
            pre_motion_3D[i] = perturb(past, eps)       # random perturbation in eps-ball
            pre_motion_3D[i].requires_grad_(True)       # set gradient tracking for inputs


        self.eval()
        for n in range(n_steps):

            # forward propagation of model
            self.set_data(data)     # transforms and sets the data
            if self.pred_type=='best_of':
                self.main(mean=False)
            elif self.pred_type=='most_likely':
                self.main(mean=True)

            # backprop of loss
            total_loss, loss_dict, loss_unweighted_dict = self.compute_loss()
            total_loss.backward()

            # gradient ascent on loss wrt input
            with torch.no_grad(): 
                for i, past in enumerate(pre_motion_3D):
                    if step_norm=='inf':
                        grad = past.grad.sign()
                    else:
                        grad = past.grad / torch.linalg.vector_norm(past.grad, ord=step_norm)

                    pre_motion_3D[i] = pre_motion_3D[i] + step_size*grad

            # project back into Lp norm ball
            if eps_norm=='inf':
                for i, past in enumerate(pre_motion_3D):
                    pre_motion_3D[i] = elementwise_clip(pre_motion_3D[i],
                                                        min=pre_motion_3D_original[i]-eps,
                                                        max=pre_motion_3D_original[i]+eps)
                    #maxpast = torch.max(pre_motion_3D[i] - pre_motion_3D_original[i])
                    #minpast = torch.min(pre_motion_3D[i] - pre_motion_3D_original[i])
                    #print('MAX:', maxpast.item())
                    #print('MIN:', minpast.item())
            else:
                for i, past in enumerate(pre_motion_3D):
                    delta = past - pre_motion_3D_original[i]
                    norm = torch.linalg.vector_norm(delta, ord=eps_norm)
                    if norm > eps:
                        delta = eps*delta/norm
                    pre_motion_3D[i] = pre_motion_3D_original[i] + delta

                    #print('norm before:', norm)
                    #delta = pre_motion_3D[i] - pre_motion_3D_original[i]
                    #norm = torch.linalg.vector_norm(delta, ord=eps_norm)
                    #print('norm afer:', norm)

            # set gradient tracking / leaf nodes
            for i, past in enumerate(pre_motion_3D):
                pre_motion_3D[i] = pre_motion_3D[i].detach().requires_grad_(True)

        return data['pre_motion_3D']



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

def elementwise_clip(x, min, max):
    return torch.max(torch.min(x, max), min)

def perturb(x, eps, eps_norm='inf'):
    return x + eps*(2*torch.rand_like(x) - 1)





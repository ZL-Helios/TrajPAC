import os
import sys
import dill
import json
import torch
import numpy as np
from tqdm import tqdm

# Trajectron++ modules
sys.path.append('./trajectron')
from trajectron.model.trajectron import Trajectron
from trajectron.model.model_registrar import ModelRegistrar

# Custom modules
from modules.trajectron.dataset_custom import Dataset_Custom

CPU = torch.device('cpu')
GPU = torch.device('cuda')


class Trajectron_Custom(Trajectron):

    def __init__(self, hyperparams, model_registrar, log_writer=None, device=CPU):
        super().__init__(model_registrar,
                         hyperparams,
                         log_writer,
                         device)
        self.ph = self.hyperparams['prediction_horizon']

    @classmethod
    def from_args(cls, args):
        DATASET = args['DeepPAC'].dataset
        DEVICE = GPU if args['DeepPAC'].gpu else CPU
        MODEL_DIR = args['Traj++'].pretrained_dir[DATASET]
        DATA_DIR = args['Traj++'].data_dir[DATASET]
        CHECKPOINT = args['Traj++'].checkpoint
        with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as config_json:
            hyperparams = json.load(config_json)

        # load trajectron model
        model_registrar = ModelRegistrar(MODEL_DIR, 'cpu')
        model_registrar.load_models(CHECKPOINT)
        trajectron = cls(hyperparams, model_registrar, device=DEVICE)

        # set environment
        env = trajectron.prepare_node_graph(DATA_DIR, hyperparams)
        trajectron.set_environment(env)     # set test dataset
        trajectron.set_annealing_params()   # set annealing params
        return trajectron


    @staticmethod
    def prepare_node_graph(model_dir, hyperparams):
        with open(model_dir, 'rb') as f:
            env = dill.load(f, encoding='latin1')
        
        if 'override_attention_radius' in hyperparams:
            for attention_radius_override in hyperparams['override_attention_radius']:
                node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                env.attention_radius[(node_type1, node_type2)] = float(attention_radius)
        
        print("-- Preparing Node Graph")
        scenes = env.scenes
        for scene in tqdm(scenes):
            scene.calculate_scene_graph(env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
        return env


    def num_parameters(self, only_trainable=False):
        raise NotImplementedError
        #return sum(p.numel() for p in self.parameters() if ((not only_trainable) or p.requires_grad))


    def predict(self, batch, future, nodeType='PEDESTRIAN', K=20, inplace=False, verbose=False):
        """
        Returns the best-of-K predicted trajectory for a given path
        batch:      batched numpy data (from dataset) needed for prediction
        future:     (B, 12, 2) true future trajectory
        node_type:  type of node we wish to predict (PEDESTRIAN, CAR, BIKE,...)
        K:          number of latent trajectories sampled
        """
        model = self.node_models_dict[nodeType]
        (first_history_index,
        x_t, y_t, x_st_t, y_st_t,
        neighbors_data_st,
        neighbors_edge_value,
        robot_traj_st_t,
        map) = batch

        # move tensors to device
        x_t = x_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        map = map.to(self.device) if map is not None else None
        robot_traj_st_t = robot_traj_st_t.to(self.device) if robot_traj_st_t is not None else None

        with torch.no_grad():
            # make prediction on mgcvae model
            predictions = model.predict(
                # inputs
                inputs = x_t,              # Input tensor including the state for each agent over time [bs, t, state].
                inputs_st = x_st_t,        # Standardized input tensor.
                # auxiliary dataset args
                first_history_indices = first_history_index,    # First timestep (index) in scene for which data is available for a node [bs]
                neighbors = neighbors_data_st,                  # Preprocessed dict (indexed by edge type) of list of neighbor states over time. [[bs, t, neighbor state]]
                neighbors_edge_value = neighbors_edge_value,    # Preprocessed edge values for all neighbor nodes [[N]]
                robot = robot_traj_st_t,                        # Standardized robot state over time. [bs, t, robot_state]
                map = map,                                      # Tensor of Map information. [bs, channels, x, y]
                # custom args
                prediction_horizon = self.ph,   # Number of prediction timesteps.
                num_samples = K,                # Number of samples from the latent space.
                z_mode = False,                 # If True: Select the most likely latent state.
                gmm_mode = False,               # If True: The mode of the GMM is sampled.
                full_dist = False,              # Samples each latent mode individually without merging them into a GMM.
                all_z_sep = False)              # Samples all latent states and merges them into a GMM as output.
            pred = predictions.cpu().detach().numpy()     # (K, B, 12, 2)

        displacement = np.sqrt(np.sum((pred - future)**2, axis=-1))     # (K, B, 12)
        displacement = np.where(np.isnan(displacement), 10*np.ones_like(displacement), displacement)

        ade = np.mean(displacement, axis=-1)    # (K, B)
        fde = displacement[:,:,-1]              # (K, B)

        idx_ade_min = np.argmin(ade, axis=0)    # (B,)
        idx_fde_min = np.argmin(fde, axis=0)    # (B,)

        pred_ade = pred[idx_ade_min, np.arange(len(idx_ade_min))]   # (B, 12, 2)
        pred_fde = pred[idx_fde_min, np.arange(len(idx_fde_min))]   # (B, 12, 2)

        if verbose:
            ade_best = ade[idx_ade_min, np.arange(len(idx_ade_min))]
            ade_best = np.mean(ade_best)
            print('Average discplacement error best: {:0.3f}'.format(ade_best))

        # apply any scaling in-place
        if inplace:
            pass
        return pred_ade
    


    def forcast(self, args, x=None, **kwargs):
        """
        black box forward propagation of Traj++ model
        args:   configuration parameters
        x:      (B, 8, 2) ndarray of all past trajectories at the given frame (optional)
        """

        # get the data dict.
        dataset = Dataset_Custom(self.env,
                                 self.hyperparams,
                                 name=args['DeepPAC'].dataset)
        idx = args['DeepPAC'].index
        data = dataset[idx]

        # prepare inputs
        if x is not None:
            batch = combine(data['past_df'], data['neighbors_df'])[0]   # (B, 8, 6)
            batch[:,:,:2] = np.copy(x)[:]                               # (B, 8, 6)
            mask = get_mask(len(batch), args['DeepPAC'].attack_scope)
            past_df = batch[:1][mask[:1]]                               # (1, 8, 6)
            nb_df = batch[1:][mask[1:]]                                 # (N, 8, 6)
        else:
            past_df = None; nb_df = None

        future = np.copy(data['future'])    # (1, 12, 2)
        batch_st = dataset.standardize_batch(data, agent_wrt=past_df, neigh_wrt=nb_df)

        # make prediction
        #self.eval()
        pred = self.predict(batch_st,       # (1, 12, 2)
                            future,
                            nodeType=args['Traj++'].node_type,
                            K=args['Traj++'].num_trajectories,
                            inplace=False,
                            **kwargs)

        # only for testing purposes. Can remove later...
        if x is not None:
            batch_raw = combine(data['past_df'], data['neighbors_df'])[0]
            assert np.all(abs(batch-batch_raw)[:,:,:2]<=args['DeepPAC'].radius+1e-4)
            assert np.all(abs(batch-batch_raw)[:,:,2:]==0)
        return pred


    def attack(self, args, type='pgd', **kwargs):

        idx = args['DeepPAC'].index
        radius = args['DeepPAC'].radius
        node_type = args['Traj++'].node_type    # 'PEDESTRIAN'
        EDGE_TYPE = ('PEDESTRIAN', 'PEDESTRIAN')
        pos_std = self.env.attention_radius[EDGE_TYPE]  # traj++ standardization std

        # get the data dict and batch
        dataset = Dataset_Custom(self.env,
                                 self.hyperparams,
                                 name=args['DeepPAC'].dataset)
        data = dataset[idx]
        batch = data['batch']


        # find adversary with pgd
        batch_adversary = self.pgd(batch,
                             n_steps=30,
                             step_size=(radius/pos_std)/20,
                             step_norm='inf',
                             eps=radius/pos_std,    # scale the radius by traj++ standardization std
                             eps_norm='inf')

        # convert from batch tuple to traj++ input
        adversary = batch2input(batch_adversary, dataset)   # (B, 8, 2)

        #original = combine(data['past_df'], data['neighbors_df'])[0][:,:,:2]   # (B, 8, 2)
        #print('original:', original)
        #print('adversary:', abs(adversary - original))
        return adversary



    def pgd(self, batch, n_steps, step_size, step_norm, eps, eps_norm):
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
        if step_norm=='inf':
            step_norm = float('inf')
        if eps_norm=='inf':
            eps_norm = float('inf')

        # unpack batch data
        (first_history_index, 
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch
        assert len(x_t) == len(x_st_t) == 1
        assert len(self.env.get_edge_types()) == 1
        EDGE_TYPE = self.env.get_edge_types()[0]        # ('PEDESTRIAN', 'PEDESTRIAN')
        neighbors_st = neighbors_data_st[EDGE_TYPE][0]  # [N, 8, 6)


        # save initial states
        x_st_original = x_st_t.detach().clone()
        nb_st_original = [nb_st.detach().clone() for nb_st in neighbors_st]

        # add random pertubation (in-place)
        x_st_t[:,:7,:2] = perturb(x_st_t, eps)[:,:7,:2] # only add noise to positions (exclude current position)
        for nb_st in neighbors_st:
            nb_st[:,:2] = perturb(nb_st, eps)[:,:2]     # only add noise to positions

        # enable gradient tracking
        x = x_t.requires_grad_(True)            # (1, 8, 6)
        x_st = x_st_t.requires_grad_(True)      # (1, 8, 6)
        for nb_st in neighbors_st:              # (8, 6)
            nb_st.requires_grad_(True)


        for n in range(n_steps):
            # loss forward/backprop
            loss = self.eval_loss(batch, 'PEDESTRIAN')
            loss.backward()
            
            with torch.no_grad():

                # filter only the positional gradients
                x_st.grad[:,:,2:] = 0
                x_st.grad[:,-1] = 0     # standardized wrt current position
                for nb_st in neighbors_st:
                    nb_st.grad[:,2:] = 0
                # normalize gradients
                if step_norm==float('inf'):
                    grad_x_st = x_st.grad.sign()
                    grad_nb_st = [nb_st.grad.sign() for nb_st in neighbors_st]
                else:
                    grad_x_st = x_st.grad / torch.linalg.vector_norm(x_st.grad, ord=step_norm)
                    grad_nb_st = [nb_st.grad / torch.linalg.vector_norm(nb_st.grad, ord=step_norm) for nb_st in neighbors_st]

                # gradient ascent on loss (in-place)
                x_st += step_size * grad_x_st
                for i, nb_st in enumerate(neighbors_st):
                    nb_st += step_size * grad_nb_st[i]  # Note: torch.no_grad() allows us to assign in-place for leaf nodes

                """
                print('\nAFTER STEP:')
                delta = x_st - x_st_original
                norm = torch.linalg.vector_norm(delta, ord=step_norm)
                print('norm agent:', norm)
                for i, nb in enumerate(neighbors_st):
                    delta = nb - nb_st_original[i]
                    norm = torch.linalg.vector_norm(delta, ord=step_norm)
                    print('norm neigh:', norm)
                """
                

                # project back into Lp norm ball (in-place)
                if eps_norm==float('inf'):
                    x_st[:] = elementwise_clip(x_st, min=x_st_original-eps, max=x_st_original+eps)
                    for i, nb_st in enumerate(neighbors_st):
                        neighbors_st[i] = elementwise_clip(nb_st, min=nb_st_original[i]-eps, max=nb_st_original[i]+eps)
                else:
                    delta = x_st - x_st_original
                    norm = torch.linalg.vector_norm(delta, ord=eps_norm)
                    if norm > eps:
                        delta = eps*delta/norm
                    x_st[:] = x_st_original + delta

                    for i, nb_st in enumerate(neighbors_st):
                        delta = nb_st - nb_st_original[i]
                        norm = torch.linalg.vector_norm(delta, ord=eps_norm)
                        if norm > eps:
                            delta = eps*delta/norm
                        neighbors_st[i] = nb_st_original[i] + delta

                """
                print('AFTER PROJECTION:')
                delta = x_st - x_st_original
                norm = torch.linalg.vector_norm(delta, ord=eps_norm)
                print('norm agent:', norm)
                for i, nb in enumerate(neighbors_st):
                    delta = nb - nb_st_original[i]
                    norm = torch.linalg.vector_norm(delta, ord=eps_norm)
                    print('norm neigh:', norm)
                """

            # enable gradient tracking
            x_st.requires_grad_(True)
            for nb_st in neighbors_st:
                nb_st.requires_grad_(True)

        return batch




# ======================================================================
# ------------------------- HELPER FUNCTIONS ---------------------------
# ======================================================================

def get_mask(bsize, attack='basic'):
    mask = np.full(bsize, True)
    if attack == 'basic':
        mask[1:] = False
    elif attack == 'env':
        mask[0] = False
    elif attack == 'full':
        pass
    return mask


def combine(listx, listy):
    """
    Combines np.ndarray lists x and y into a single batch
    x:  list of np arrays. e.g. (B, 8, 6)
    y:  list of np arrays. e.g. [(N1, 8, 6)...(Nb, 8, 6)]
    Returns a list [(N1+1, 8, 6)...(Nb+1, 8, 6)]
    """
    assert len(listx)==len(listy)
    xy = []
    for i in range(len(listx)):
        x = listx[i]    # (*, 8, 6)
        y = listy[i]    # (Ni, *, 8, 6)
        # broadcast smaller array to same shape as larger one
        if len(x.shape) > len(y.shape):
            y = np.broadcast_to(y, x.shape[1:])
            y = np.expand_dims(y, axis=0)
        elif len(y.shape) > len(x.shape):
            x = np.broadcast_to(x, y.shape[1:])
            x = np.expand_dims(x, axis=0)
        else:
            raise NotImplementedError
        z = np.concatenate((x, y), axis=0)  # (Ni+1, 8, 6)
        xy.append(z)
    return xy


def elementwise_clip(x, min, max):
    return torch.max(torch.min(x, max), min)

def perturb(x, eps, eps_norm=float('inf')):
    if eps_norm==float('inf'):
        return x + eps*(2*torch.rand_like(x) - 1)
    else:
        raise NotImplementedError


def batch2input(batch, dataset):
    """
    Convert the batch collection into the input trajectories. 
        batch:      data tuple returned by the dataloader
    The input trajectories consist of the following arrays:
        past:       (8, 2) past agent trajectory
        neighbors:  (N, 8, 2) past neighbor trajectories
    returns (B=N+1, 8, 2) input array
    """

    (first_history_index, 
     x_t, y_t, x_st_t, y_st_t,
     neighbors_data_st,
     neighbors_edge_value,
     robot_traj_st_t,
     map) = batch

    EDGE_TYPE = ('PEDESTRIAN', 'PEDESTRIAN')
    nb_st_t = neighbors_data_st[EDGE_TYPE][0]

    # get numpy versions
    x = x_t[0].detach().numpy()                         # (8, 6)
    x_st = x_st_t[0].detach().numpy()                   # (8, 6)
    nb_st = [nb.detach().numpy() for nb in nb_st_t]     # [N, 8, 6)

    # unstandardize trajectories
    agent = dataset.unstandardize_agent(x_st, mean=x[-1,:2])[np.newaxis]    # (1, 8, 6)
    neighbors = dataset.unstandardize_neighbor(nb_st, mean=x[-1])           # [(N, 8, 6)]
    #print('AGENT:', agent)
    #print('NEIGH:', neighbors)
    return combine(agent, neighbors)[0][:,:,:2]                             # (B, 8, 2)
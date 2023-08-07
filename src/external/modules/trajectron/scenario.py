# NOTE: The current working directory when executing this module should be in the 
#       Trajectron++ project workspace. It is not in the DeepPAC workspace!

import os
import math
import sys
import numpy as np
import torch
import dill
import json
from tqdm import tqdm

# Trajectron++ modules
sys.path.append('./trajectron')
from trajectron.model.model_registrar import ModelRegistrar

# Custom modules
from modules.trajectron.model_custom import Trajectron_Custom
from modules.trajectron.dataset_custom import Dataset_Custom

CPU = torch.device('cpu')
GPU = torch.device('cuda')


class Scenario():
    """
    Class that acts like a dataloader for generating scenario data (i.e., noisy data about
    the center of L-inf B(x̂,r)). The dataset is generated based on Trajectron++, and the
    center point x̂ can be selected by indexing. 
    """
    def __init__(self, args):
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

        # define Traj++ parameters
        MODEL_DIR = args['Traj++'].pretrained_dir[DATASET]
        DATA_DIR = args['Traj++'].data_dir[DATASET]
        CHECKPOINT = args['Traj++'].checkpoint
        with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as config_json:
            self.hyperparams = json.load(config_json)

        # prepare the graph data
        self.env = self._prepare_node_graph(DATA_DIR)

        self.dataset = self._load_dataset()
        self.trajectron = self._load_model(MODEL_DIR, CHECKPOINT)
        #self.trajectron.eval()

        # NOTE zara1
        # num parameters: 127654

        # just for testing purposes (can delete after if no issues)
        model = self.trajectron.node_models_dict[self.args['Traj++'].node_type]
        model_et = model.edge_types
        et = self.env.get_edge_types()
        assert et == model_et

    def _prepare_node_graph(self, data_dir):
        with open(data_dir, 'rb') as f:
            env = dill.load(f, encoding='latin1')
        
        if 'override_attention_radius' in self.hyperparams:
            for attention_radius_override in self.hyperparams['override_attention_radius']:
                node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                env.attention_radius[(node_type1, node_type2)] = float(attention_radius)
        
        print("-- Preparing Node Graph")
        scenes = env.scenes
        for scene in tqdm(scenes):
            scene.calculate_scene_graph(env.attention_radius,
                                        self.hyperparams['edge_addition_filter'],
                                        self.hyperparams['edge_removal_filter'])
        return env


    def _load_dataset(self):
        dataset = Dataset_Custom(self.env, self.hyperparams, name=self.args['DeepPAC'].dataset)
        return dataset


    def _load_model(self, model_dir, checkpoint=100):
        model_registrar = ModelRegistrar(model_dir, 'cpu')
        model_registrar.load_models(checkpoint)
        trajectron = Trajectron_Custom(self.hyperparams,
                                       model_registrar,
                                       device=self.device)
        trajectron.set_environment(self.env)    # load test dataset
        trajectron.set_annealing_params()       # set annealing params
        return trajectron


    def __getitem__(self, idx):
        print(f'generating noisy data around sample {idx}...')

        # Get the batch data containing the given idx=(fid,pid) pair
        data = self.dataset[idx]
        batch = combine(data['past_df'], data['neighbors_df'])[0]   # (B, 8, 6) *concat [(1,8,6)] with [(N,8,6)]
        mask = get_mask(len(batch), self.attack_scope)               # (B,) bool mask


        # Get the center point x̂ of the L-inf ball B(x̂,r)
        past = np.copy(data['past'])        # (1, 8, 2)     past traj position
        future = np.copy(data['future'])    # (1, 12, 2)    future traj position
        pred = self.predict(data,           # (1, 12, 2)
                            future,
                            st_wrt=None,
                            st_nb_wrt=None,
                            K=self.args['Traj++'].num_trajectories,
                            nodeType=self.args['Traj++'].node_type,
                            inplace=True,       # whether to apply in-place standardization of future
                            verbose=True)
        print('\tBatch: ', batch.shape)
        print('\tPast:  ', past.shape)
        print('\tFuture:', future.shape)
        #assert np.max(abs(future-data['future']))>0


        # Generate scenario optimization samples
        scenarios = self.create_scenarios(batch, self.radius)   # (T1+T2+T3, B, 8, 6)
        phases = self.split_into_phases(scenarios)              # [(T1, B, 8, 6)...]
        assert np.all((scenarios[0] - batch)[:,:,:2] < self.radius)


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
                                     st_nb_wrt=nb_df,   # TODO: what happens when nb_df is only a subset of total nbs
                                     K=self.args['Traj++'].num_trajectories,
                                     nodeType=self.args['Traj++'].node_type,
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




    def predict(self, data, future, st_wrt=None, st_nb_wrt=None, K=20, nodeType='PEDESTRIAN', inplace=False, verbose=False):        
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
                                           nb_wrt=st_nb_wrt,
                                           max_ht=self.hyperparams['maximum_history_length'],
                                           max_ft=self.hyperparams['prediction_horizon'])
        pred = self.trajectron.predict(batch,
                                       future,
                                       nodeType=nodeType,
                                       K=K,
                                       inplace=inplace,
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
        mask = get_mask(len(batch), self.attack_scope)

        # make prediction
        future = np.copy(data['future'])    # (1, 12, 2)
        past_df = batch[:1][mask[:1]]       # (1, 8, 6)
        nb_df = batch[1:][mask[1:]]         # (N, 8, 6)
        pred = []
        for i in range(num_predictions):
            recon = self.predict(data,      # (1, 12, 2)
                                future,
                                st_wrt=past_df,
                                st_nb_wrt=nb_df,
                                K=self.args['Traj++'].num_trajectories,
                                nodeType=self.args['Traj++'].node_type,
                                inplace=False,
                                **kwargs)
            pred.append(recon)
        pred = np.stack(pred)

        batch_raw = combine(data['past_df'], data['neighbors_df'])[0]
        assert np.all(abs(batch-batch_raw)[:,:,:2]<=self.radius+1e-4)
        assert np.all(abs(batch-batch_raw)[:,:,2:]==0)
        return pred


    def create_scenarios(self, x, radius):
        """
        Function to generate n iid uniform samples within B(x̂,r) for scenario optimization
        x:              (B, *...) center of the L-inf ball (B, 8, 6)
        radius:         radius of the L-inf ball B(x̂,r)
        num_scenarios:  number of scenarios/samples
        """
        num_scenarios = self.FThreshold + self.SThreshold + self.TThreshold
        radius = radius*np.ones_like(x)
        radius[:,:,2:] = 0  # only add noise to the position data
        radius[x==0.0] = 0  # ignore padded positions
        x_l = x - radius
        x_u = x + radius
        return (x_u-x_l)*np.random.rand(num_scenarios, *x.shape) + x_l

    def split_into_phases(self, x):
        """ Function that splits (T1+T2+T3, B, 8, 2) into a list of data for 3 phases """
        y = x.copy()
        del x   # delete reference to original data
        return [
            y[:self.FThreshold],
            y[self.FThreshold:self.FThreshold + self.SThreshold],
            y[self.FThreshold + self.SThreshold:]
        ]





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


def to(container, device=CPU, keys=None):
    # TODO : make into recursive func?
    if type(container)==dict:
        keys = container.keys() if keys is None else keys
        for key in keys:
            obj = container[key]
            if obj is not None and type(obj) == torch.Tensor:
                container[key] = container[key].to(device)
    elif type(container)==list:
        for i in range(len(container)):
            obj = container[i]
            if obj is not None and type(obj) == torch.Tensor:
                container[i] = container[i].to(device)


def print_container_shapes(obj):
    def get_container_shapes(obj, depth=0):
        res = ''
        # base case
        if type(obj) == torch.Tensor or type(obj) == np.ndarray:
            return depth*'\t' + f'{obj.shape}'
        
        # recursive cases
        elif type(obj) == list:
            raise NotImplementedError

        elif type(obj) == dict:
            res += depth*'\t' + '{'
            for i,key in enumerate(obj):    
                res += f'{key}:\n' + get_container_shapes(obj[key], depth+1)
                if i < len(obj)-1:
                    res += ',\n' + depth*'\t'
            res += '}'

        return res
    print(get_container_shapes(obj, depth=0))



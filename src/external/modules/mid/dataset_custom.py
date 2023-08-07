import torch
import numpy as np

# MID modules
from dataset import collate, get_timesteps_data
from dataset.preprocessing import get_relative_robot_traj


class Dataset_Custom(object):
    """
    Custom dataset class that defines the __getitem__ function to allow us to index 
    specific preprocessed members of the dataset. The data is in the form of torch.Tensor.
    """

    def __init__(self, env, dataset, hyperparams, config) -> None:
        super().__init__()
        self.env = env
        self.scenes = env.scenes
        self.dataset = dataset
        self.hyperparams = hyperparams
        self.config = config
        self.scene_graphs = []      # current scene graphs at this batch / iterator
        self.idx_map = self._get_idx_map()

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):

        if type(idx)==int:
            scene_id, t = self.idx_map[idx]
            scene = self.scenes[scene_id]
            ts = np.arange(t, t+10)
            ts, nodes, scene_graphs = self.get_graph_data_batched(scene, ts,
                                                                  nodeType='PEDESTRIAN',
                                                                  min_ht=7,
                                                                  max_ft=12)
            # set current scene attributes
            self.scene = scene
            self.scene_graphs = scene_graphs
            # get batch data
            batch = self.get_batch(scene, ts, nodes, scene_graphs, wrt=None)
            return batch, ts, nodes, scene, scene_graphs

        elif type(idx)==tuple:
            fid, pid = idx
            scene_id = self.config.dataset+'_test'
            scene, ts, nodes, nb_nodes, scene_graphs = self.get_graph_data(scene_id,
                                                                           fid,
                                                                           pid,
                                                                           bsize=1)
            if len(ts)==0:
                raise Exception(f'The given (fid={fid},pid={pid}) pair is not valid!')
            past_df, future_df, neighbors_df = get_state_data(ts,
                                                              nodes,
                                                              nb_nodes,
                                                              states=['position', 'velocity', 'acceleration'],
                                                              past_horizon=7,
                                                              pred_horizon=12)
            batch = self.get_batch(scene,
                                   ts,
                                   nodes,
                                   scene_graphs,
                                   wrt=None)

            # create the fids/pids arrays corresponding to the batch data
            fids, pids = [], []
            max_ht = self.hyperparams['maximum_history_length']+1
            for i,t in enumerate(ts):
                fids_t, pids_t = [], []
                node = nodes[i]
                fids_t.append(np.array([t2fid(scene_id, t)//10-7+k for k in range(max_ht)]))
                pids_t.append(int(node.id))
                for neighbor in nb_nodes[i]:
                    fids_t.append(np.array([t2fid(scene_id, t)//10-7+k for k in range(max_ht)]))
                    pids_t.append(int(neighbor.id))
                fids.append(np.stack(fids_t))   # (B, 8)
                pids.append(np.stack(pids_t))   # (B,)
            fids = np.stack(fids)   # (ts, B, 8)
            pids = np.stack(pids)   # (ts, B)

            data = {
                'batch': batch,                 # batched data for each node at corresponding timestep
                'past': past_df[:,:,:2],        # (B, 8, 2) past traj
                'future': future_df[:,:,:2],    # (B, 12, 2) future traj
                'past_df': past_df,
                'future_df': future_df,
                'neighbors_df': neighbors_df,
                'scene': scene,
                'scene_graphs': scene_graphs,
                'timesteps': ts,
                'nodes': nodes,
                'fids': fids[0],    # (B, 8) NOTE : can only plot one sample at a time
                'pids': pids[0],    # (B)
            }
            return data



    def _get_idx_map(self):
        valid_mapping = []
        for i,scene in enumerate(self.scenes):
            for t in range(0, scene.timesteps, 10):
                ts = np.arange(t, t+10)
                batch = get_timesteps_data(
                    env=self.env,
                    scene=scene,
                    t=ts,
                    node_type='PEDESTRIAN',
                    state=self.hyperparams['state'],                # position
                    pred_state=self.hyperparams['pred_state'],      # velocity
                    edge_types=self.env.get_edge_types(),
                    min_ht=7,
                    max_ht=self.hyperparams['maximum_history_length'],
                    min_ft=12,
                    max_ft=12,
                    hyperparams=self.hyperparams)
                if batch is not None:
                    valid_mapping.append((i, t))
        return valid_mapping



    def get_graph_data(self, scene_id, fid, pid, bsize=1, nodeType='PEDESTRIAN', min_ht=7, max_ft=12):
        """
        Returns the graph data according to the scene_id, frame_id, and path_id
        """
        # get the scene
        scene = None
        for _scene in self.scenes:
            if _scene.name == scene_id:
                scene = _scene
        if scene is None:
            return None
        
        # get the graph data corresponding to the fid / pid
        t = fid2t(scene.name, fid)      # convert fid to scene timestep 
        ts = np.arange(t, t+bsize)
        nodes_per_ts = scene.present_nodes(ts,
                                           type=nodeType,
                                           min_history_timesteps=min_ht,
                                           min_future_timesteps=max_ft,
                                           return_robot=not self.hyperparams['incl_robot_node'])
        timesteps = []
        nodes = []
        nb_nodes = []
        scene_graphs = []
        for t in nodes_per_ts:
            scene_graph = scene.get_scene_graph(t,
                                                self.env.attention_radius,
                                                self.hyperparams['edge_addition_filter'],
                                                self.hyperparams['edge_removal_filter'])
            for node in nodes_per_ts[t]:
                if node.id == str(pid):
                    timesteps.append(t)
                    nodes.append(node)
                    scene_graphs.append(scene_graph)

                    # get neighbor data
                    for edge_type in self.env.get_edge_types():
                        connected_nodes = scene_graph.get_neighbors(node, edge_type[-1])
                        nb_nodes.append(connected_nodes)

        return scene, timesteps, nodes, nb_nodes, scene_graphs




    def get_graph_data_batched(self, scene, ts, nodeType, min_ht=7, max_ft=12):
        nodes_per_ts = scene.present_nodes(ts,
                                           type=nodeType,
                                           min_history_timesteps=min_ht,
                                           min_future_timesteps=max_ft,
                                           return_robot=not self.hyperparams['incl_robot_node'])
        batch_nodes = []
        batch_ts = []
        batch_scene_graphs = []
        for t in nodes_per_ts:
            scene_graph = scene.get_scene_graph(t,
                                                self.env.attention_radius,
                                                self.hyperparams['edge_addition_filter'],
                                                self.hyperparams['edge_removal_filter'])
            nodes = nodes_per_ts[t]
            for node in nodes:
                batch_nodes.append(node)
                batch_ts.append(t)
                batch_scene_graphs.append(scene_graph)
        return batch_ts, batch_nodes, batch_scene_graphs



    def get_batch(self, scene, timesteps, nodes, scene_graphs=None, wrt=None, nb_wrt=None):
        """
        Returns a batch of inputs for each node at timestep. 
        The batch is standardized with respect to the past traj (wrt)
        timesteps:      (B,) timesteps of each node
        nodes:          (B,) nodes at each timestep
        scene_graphs:   (B,<scene_graph>) scene graphs at each node/timestep pair
        wrt:            (B, 8, 6) numpy array. (optional) 
                        If given the batch data is standardized with respect to this tensor.
        nb_wrt:         [(Ni, 8, 6)]_i=1..B list of numpy arrays. (optional)
                        Standardize the N_i neighbors according to the paths for each of B trajectories
        """
        assert len(timesteps) == len(nodes)
        if wrt is not None:
            assert len(timesteps) == len(wrt)

        STATE = self.hyperparams['state']               # {'PEDESTRIAN': {'position': ['x', 'y'], 'velocity': [..], 'acceleration': [..]}}
        PRED_STATE = self.hyperparams['pred_state']     # {'PEDESTRIAN': {'velocity': ['x', 'y']}}
        max_ht = self.hyperparams['maximum_history_length']
        max_ft = 12

        batch = []
        for i in range(len(timesteps)):
            # get node/timestep data
            t = timesteps[i]
            node = nodes[i]
            if scene_graphs:
                scene_graph = scene_graphs[i]
            else:
                scene_graph = scene.get_scene_graph(t,
                                                    self.env.attention_radius,
                                                    self.hyperparams['edge_addition_filter'],
                                                    self.hyperparams['edge_removal_filter'])
            # get state data
            ts_range_x = np.array([t-max_ht, t])
            ts_range_y = np.array([t+1, t+max_ft])

            #if wrt is not None:
            #    x = node.get(ts_range_x, STATE[node.type])
            #    d = np.max(abs(x-wrt[i]))
            #    print(abs(x-wrt[i]))
            #    assert d < 0.1

            x = node.get(ts_range_x, STATE[node.type]) if wrt is None else wrt[i]   # (8, 6)
            y = node.get(ts_range_y, PRED_STATE[node.type])                         # (12, 2)
            first_history_index = (max_ht - node.history_points_at(t)).clip(0)

            # standardize the state data
            mean, std = self.env.get_standardize_params(STATE[node.type], node.type)
            std[0:2] = self.env.attention_radius[(node.type, node.type)]
            rel_state = np.zeros_like(x[0])
            rel_state[0:2] = np.array(x)[-1, 0:2]
            x_st = self.env.standardize(x, STATE[node.type], node.type, mean=rel_state, std=std)
            if next(iter(PRED_STATE[node.type])) == 'position':
                y_st = self.env.standardize(y.numpy(), PRED_STATE[node.type], node.type, mean=rel_state[0:2])
            else:
                y_st = self.env.standardize(y, PRED_STATE[node.type], node.type)
            
            x_t = torch.tensor(x, dtype=torch.float)
            y_t = torch.tensor(y, dtype=torch.float)
            x_st_t = torch.tensor(x_st, dtype=torch.float)
            y_st_t = torch.tensor(y_st, dtype=torch.float)


            # Neighbor data
            if self.hyperparams['edge_encoding']:
                neighbors_st_dict = dict()
                neighbors_edge_value = dict()

                for edge_type in self.env.get_edge_types():
                    connected_nodes = scene_graph.get_neighbors(node, edge_type[1])
                    neighbors_st_dict[edge_type] = list()

                    # get the edge masks for the current node at the current timestep
                    if self.hyperparams['dynamic_edges'] == 'yes':
                        edge_masks = torch.tensor(scene_graph.get_edge_scaling(node), dtype=torch.float)
                        neighbors_edge_value[edge_type] = edge_masks

                    # get the neighbor states
                    for j,connected_node in enumerate(connected_nodes):

                        #if nb_wrt is not None:
                        #    nb = connected_node.get(np.array([t-max_ht, t]), STATE[connected_node.type], padding=0.0)
                        #    d = np.max(abs(nb - nb_wrt[i][j]))
                        #    print('FLAG:', abs(nb - nb_wrt[i][j]))
                        #    assert d < 0.1


                        neighbor_state_np = connected_node.get(np.array([t-max_ht, t]),
                                                               STATE[connected_node.type],
                                                               padding=0.0) if nb_wrt is None else nb_wrt[i][j]
                        # standardize the neighbor state data
                        mean, std = self.env.get_standardize_params(STATE[connected_node.type], node_type=connected_node.type)
                        std[0:2] = self.env.attention_radius[edge_type]
                        equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                        rel_state = np.zeros_like(neighbor_state_np)
                        rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                        neighbor_state_np_st = self.env.standardize(neighbor_state_np,
                                                                    STATE[connected_node.type],
                                                                    node_type=connected_node.type,
                                                                    mean=rel_state,
                                                                    std=std)
                        neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                        neighbors_st_dict[edge_type].append(neighbor_state)

            # Robot data
            robot_traj_st_t = None
            ts_range_r = np.array([t, t+max_ft])
            if self.hyperparams['incl_robot_node']:
                x_node = node.get(ts_range_r, STATE[node.type])
                if scene.non_aug_scene is not None:
                    robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
                else:
                    robot = scene.robot
                robot_type = robot.type
                robot_traj = robot.get(ts_range_r, STATE[robot_type], padding=0.0)
                robot_traj_st_t = get_relative_robot_traj(self.env, self.state, x_node, robot_traj, node.type, robot_type)

            # Map
            map_tuple = None
            if self.hyperparams['use_map_encoding']:
                if node.type in self.hyperparams['map_encoder']:
                    if node.non_aug_node:
                        x = node.non_aug_node.get(np.array([t]), STATE[node.type])
                    me_hyp = self.hyperparams['map_encoder'][node.type]
                    if 'heading_state_index' in me_hyp:
                        heading_state_index = me_hyp['heading_state_index']
                        # We have to rotate the map in the opposite direction of the agent to match them
                        if type(heading_state_index) is list:  # infer from velocity or heading vector
                            heading_angle = -np.arctan2(x[-1, heading_state_index[1]],
                                                        x[-1, heading_state_index[0]]) * 180 / np.pi
                        else:
                            heading_angle = -x[-1, heading_state_index] * 180 / np.pi
                    else:
                        heading_angle = None
                    scene_map = scene.map[node.type]
                    map_point = x[-1, :2]
                    patch_size = self.hyperparams['map_encoder'][node.type]['patch_size']
                    map_tuple = (scene_map, map_point, heading_angle, patch_size)

            sample = (first_history_index,
                      x_t, y_t, x_st_t, y_st_t,
                      neighbors_st_dict, neighbors_edge_value, 
                      robot_traj_st_t,
                      map_tuple)
            batch.append(sample)

        return collate(batch)


    def standardize_batch(self, data, st_agent=None, st_neigh=None):
        """
        constructs the batch data to be inputted into the model.predict() method.
        This batch data is standardized with respect to the agent 'agent_wrt' and neighbors 'neigh_wrt'
        data:       dictionary returned by __getitem__()
        agent_wrt:  (1, 8, 6) numpy array (optional)
        neigh_wrt:  (N, 8, 6) numpy array (optional)
        """
        # set empty arrays to None
        st_agent = None if (st_agent is not None and len(st_agent) == 0) else st_agent
        st_neigh = None if (st_neigh is not None and len(st_neigh) == 0) else st_neigh
        # batch the neighbor nodes for input into get_batch
        if st_neigh is not None: st_neigh = [st_neigh]   # (1, N, 8, 6)

        if st_agent is None and st_neigh is None:
            batch = data['batch']
        else:
            batch = self.get_batch(scene=data['scene'],
                                   timesteps=data['timesteps'],
                                   nodes=data['nodes'],
                                   scene_graphs=data['scene_graphs'],
                                   wrt=st_agent,
                                   nb_wrt=st_neigh)
        return batch







# ======================================================================
# ------------------------- HELPER FUNCTIONS ---------------------------
# ======================================================================

def fid2t(scene_id, fid):
    if scene_id == 'eth_test':
        t = (fid//10) - 78
    elif scene_id == 'hotel_test':
        t = (fid//10) - 0
    elif scene_id == 'univ_test':
        t = (fid//10) - 0
    elif scene_id == 'zara1_test':
        t = (fid//10) - 0
    elif scene_id == 'zara2_test':
        t = (fid//10) - 1
    return t

def t2fid(scene_id, t):
    if scene_id == 'eth_test':
        fid = 10*(t+78)
    elif scene_id == 'hotel_test':
        fid = 10*(t+0)
    elif scene_id == 'univ_test':
        fid = 10*(t+0)
    elif scene_id == 'zara1_test':
        fid = 10*(t+0)
    elif scene_id == 'zara2_test':
        fid = 10*(t+1)
    return fid


def get_state_data(ts, nodes, nb_nodes=None, states=None, past_horizon=7, pred_horizon=12):
    """
    Returns the trajectory data (past and future) for all nodes at all timesteps
    ts:     list of timesteps
    nodes:  list of Node corresponding to each timestep
    states: list of states [position, velocity, acceleration] whose data we wish to get
    """
    assert len(ts)==len(nodes)
    STATE = dict()  # {'position': ['x', 'y'], 'velocity': ['x', 'y'], 'acceleration': ['x', 'y']}
    states = ['position', 'velocity', 'acceleration'] if states is None else states    
    for state in states:
        STATE[state] = ['x', 'y']
    hist = past_horizon
    futr = pred_horizon

    past_batch = []
    future_batch = []
    neighbors_batch = []
    for i in range(len(ts)):
        t, node = ts[i], nodes[i]
        past = node.get(np.array([t-hist, t]), STATE)
        past = past[~np.isnan(past.sum(axis=1))]
        future = node.get(np.array([t+1, t+futr]), STATE)
        future = future[~np.isnan(future.sum(axis=1))]
        past_batch.append(past)
        future_batch.append(future)

        if nb_nodes is not None:
            neighbors = []
            for neighbor in nb_nodes[i]:
                neighbors.append(neighbor.get(np.array([t-hist, t]), STATE, padding=0.0))
            neighbors_batch.append(np.stack(neighbors))

    past_batch = np.stack(past_batch)                   # (B, 8, 6)
    future_batch = np.stack(future_batch)               # (B, 12, 6)
    return past_batch, future_batch, neighbors_batch    # [(Ni, 8, 6)...]

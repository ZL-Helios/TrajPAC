import torch
import numpy as np

# Trajectron++ modules
from trajectron.model.dataset.preprocessing import get_timesteps_data, collate, get_relative_robot_traj


class Dataset_Custom(object):
    
    def __init__(self, env, hyperparams, name):
        super().__init__()
        self.env = env
        self.hyperparams = hyperparams
        self.name = name

    def __getitem__(self, idx):
        if type(idx) == int:
            raise(NotImplementedError)

        elif type(idx) == tuple:
            fid, pid = idx
            scene_id = self.name+'_test'
            scene, ts, nodes, nb_nodes, scene_graphs = self.get_graph_data(scene_id,
                                                                           fid,
                                                                           pid,
                                                                           bsize=1) # NOTE : must be 1 in our case
            if len(ts)==0:
                raise Exception(f'The given (fid={fid},pid={pid}) pair is not valid!')
            past, future, neighbors = get_state_data(ts,
                                                     nodes,
                                                     nb_nodes,
                                                     states=['position', 'velocity', 'acceleration'],
                                                     past_horizon=self.hyperparams['maximum_history_length'],   # 7
                                                     pred_horizon=self.hyperparams['prediction_horizon'])       # 12
            batch = self.get_batch(scene,
                                   ts,
                                   nodes,
                                   scene_graphs,
                                   wrt=None,
                                   nb_wrt=None,
                                   max_ht=self.hyperparams['maximum_history_length'],
                                   max_ft=self.hyperparams['prediction_horizon'])
            
            
            # create the fids/pids arrays corresponding to the batch data
            fids, pids = [], []
            nts = self.hyperparams['maximum_history_length']+1
            for i,t in enumerate(ts):
                fids_t, pids_t = [], []
                node = nodes[i]
                pids_t.append(int(node.id))
                fids_t.append(np.array([t2fid(scene_id, t)//10-7+k for k in range(nts)]))
                for nb_node in nb_nodes[i]:
                    pids_t.append(int(nb_node.id))
                    fids_t.append(np.array([t2fid(scene_id, t)//10-7+k for k in range(nts)]))
                fids.append(np.stack(fids_t))
                pids.append(np.stack(pids_t))
            fids = np.stack(fids)   # (ts, B, 8)
            pids = np.stack(pids)   # (ts, B)


            data = {
                'batch': batch,                 # batched data for each node at corresponding timestep
                'past': past[:,:,:2],           # (B, 8, 2) past traj
                'future': future[:,:,:2],       # (B, 12, 2) future traj
                'past_df': past,
                'future_df': future,
                'neighbors_df': neighbors,
                'scene': scene,
                'scene_graphs': scene_graphs,
                'timesteps': ts,
                'nodes': nodes,
                'fids': fids[0],    # (B, 8) NOTE: all scaled by /10
                'pids': pids[0],    # (B)
            }
            return data


    
    def get_graph_data(self, scene_id, fid, pid, bsize=1, nodeType='PEDESTRIAN', min_ht=7, min_ft=12):
        """
        Returns the timesteps, nodes, and graphs for the given scene, fid, and pid
        """
        # get scene
        for _scene in self.env.scenes:
            if _scene.name == scene_id:
                scene = _scene

        # get graph data
        timesteps = []
        nodes = []
        nb_nodes = []
        scene_graphs = []
        t = fid2t(scene_id, fid) 
        ts = np.arange(t, t + bsize)
        nodes_per_ts = scene.present_nodes(ts,
                                           type=nodeType,
                                           min_history_timesteps=min_ht,
                                           min_future_timesteps=min_ft,
                                           return_robot=not self.hyperparams['incl_robot_node'])
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



    def get_batch(self, scene, timesteps, nodes, scene_graphs=None, wrt=None, nb_wrt=None, max_ht=7, max_ft=12):
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

        batch = []
        for i,t in enumerate(timesteps):
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


    def standardize_batch(self, data, agent_wrt=None, neigh_wrt=None):
        """
        constructs the batch data to be inputted into the predict() method.
        This batch data is standardized with respect to the agent 'agent_wrt' and neighbors 'neigh_wrt'
        data:       dictionary returned by __getitem__()
        agent_wrt:  (1, 8, 6) numpy array (optional)
        neigh_wrt:  (N, 8, 6) numpy array (optional)
        """
        # set empty arrays to None
        agent_wrt = None if (agent_wrt is not None and len(agent_wrt) == 0) else agent_wrt
        neigh_wrt = None if (neigh_wrt is not None and len(neigh_wrt) == 0) else neigh_wrt
        # batch the neighbor nodes for input into get_batch
        if neigh_wrt is not None: neigh_wrt = [neigh_wrt]   # (1, N, 8, 6)

        if agent_wrt is None and neigh_wrt is None:
            batch = data['batch']
        else:
            batch = self.get_batch(scene=data['scene'],
                                   timesteps=data['timesteps'],
                                   nodes=data['nodes'],
                                   scene_graphs=data['scene_graphs'],
                                   wrt=agent_wrt,
                                   nb_wrt=neigh_wrt,
                                   max_ht=self.hyperparams['maximum_history_length'],
                                   max_ft=self.hyperparams['prediction_horizon'])
        return batch


    def get_valid_samples(self):
        valid_samples = []
        scenes = self.env.scenes
        scene_id = 0
        ts = 0
        while scene_id < len(scenes):
            if ts < scenes[scene_id].timesteps:
                batch = get_timesteps_data(
                    env=self.env,
                    scene=scenes[scene_id],
                    t=np.arange(ts, ts+10),                                 # timestep range in scene
                    node_type=self.node_type,                               # node type of nodes for which the data shall be pre-processed
                    state=self.hyperparams['state'],                        # specification of the node state
                    pred_state=self.hyperparams['pred_state'],              # specification of the prediction state
                    edge_types=self.edge_types,                             # list of all edge types for which neighbors are pre-processed
                    min_ht=7,                                               # minimum history timesteps
                    max_ht=self.hyperparams['maximum_history_length'],      # maximum history timesteps
                    min_ft=12,                                              # (unused)
                    max_ft=12,                                              # maximum future timesteps (prediction horizon)
                    hyperparams=self.hyperparams)                           # model hyperparams
                if batch:
                    valid_samples.append((scene_id, ts))
                ts += 10
            else:
                scene_id += 1
                ts = 0
        return valid_samples

    def idx_2_scene_and_frame(self, idx):
        scene_id = 0
        frame_id = 10*idx
        scenes = self.env.scenes
        while frame_id > scenes[scene_id].timesteps:
            frame_id -= 10*(scenes[scene_id].timesteps//10)
            scene_id += 1
        return scene_id, min(frame_id, scenes[scene_id].timesteps-10)




    def unstandardize_agent(self, x, mean, std=None):
        """
        Returns the unstandardized array x * std + mean
        x:      (8, 6) standardized np array
        mean:   (6,) array of means
        std:    (6,) array of stds
        """
        assert x.shape == (8,6)
        STATE = self.hyperparams['state']

        rel_state = np.zeros_like(x[-1])        # (6,)
        rel_state[:mean.size] = np.array(mean)  # overwrites first n=mean.size elems
        if std is None:
            _, std = self.env.get_standardize_params(STATE['PEDESTRIAN'], 'PEDESTRIAN')
            std[0:2] = self.env.attention_radius[('PEDESTRIAN', 'PEDESTRIAN')]
        return self.env.unstandardize(x, state=STATE['PEDESTRIAN'], node_type='PEDESTRIAN', mean=rel_state, std=std)



    def unstandardize_neighbor(self, neighbors, mean, std=None):
        """
        Returns the list of standardized neighbor states
        neighbors:  [N, 8, 6) list of N standardized np array neighbor states
        mean:       (6,) array of means
        std:        (6,) array of stds
        returns [(N, 8, 6)] list of N neighbor states
        """
        assert neighbors[0].shape == (8,6)
        STATE = self.hyperparams['state']

        out = []
        for x in neighbors:

            # get mean and std
            rel_state = np.zeros_like(x)                # (8, 6)
            rel_state[:, :mean.size] = np.array(mean)   # overwrites all elems
            if std is None:
                _, std = self.env.get_standardize_params(STATE['PEDESTRIAN'], 'PEDESTRIAN')
                std[0:2] = self.env.attention_radius[('PEDESTRIAN', 'PEDESTRIAN')]
            
            # unstandardize neighbor state
            x_st = self.env.unstandardize(x, state=STATE['PEDESTRIAN'], node_type='PEDESTRIAN', mean=rel_state, std=std)
            out.append(x_st)    # (8, 6)
        
        return [np.array(out)]  # [(N, 8, 6)]






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
    Returns the trajectory data (past, future, neigbors) for all nodes at all timesteps
    ts:             list of timesteps
    nodes:          list of Node corresponding to each timestep
    nb_nodes:       list of list of neighbor nodes for the given timestep and node
    states:         list of states [position, velocity, acceleration] whose data we wish to get
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
            nb_past = []
            for neighbor in nb_nodes[i]:
                nb_past.append(neighbor.get(np.array([t-hist, t]), STATE, padding=0.0))
            neighbors_batch.append(np.stack(nb_past))   # (Ni, 8, 6)

    past_batch = np.stack(past_batch)                   # (B, 8, 6)
    future_batch = np.stack(future_batch)               # (B, 12, 6)
    return past_batch, future_batch, neighbors_batch    # [(Ni, 8, 6)...(Nb, 8, 6)]

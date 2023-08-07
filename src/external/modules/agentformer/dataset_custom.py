import os
import numpy as np

# AgentFormer modules
from utils.config import Config
from data.dataloader import data_generator

# Custom modules
#from modules.agentformer.utils import *


class Dataset_Custom(data_generator):
    
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')
        super(Dataset_Custom, self).__init__(cfg,
                                             self.log,
                                             split='test',
                                             phase='testing')

    def __getitem__(self, idx):
        fid, pid = idx
        scene = self.getScene(dset2scene(self.cfg.dataset))
        try:
            data = scene(fid)   # NOTE: data is scaled! by /= cfg.traj_scale
        except:
            raise Exception(f'The given (fid={fid},pid={pid}) pair is not valid!')

        data['fids'] = np.array([[fid-(7-i) for i in range(8)] for pid in data['valid_id']])    # (B, 8)
        data['pids'] = np.array(data['valid_id'])   # (B,)
        return data
        data = {
            'pre_motion_3D': pre_motion_3D,         # (B, 8, 2)
            'fut_motion_3D': fut_motion_3D,         # (B, 12, 2)
            'fut_motion_mask': fut_motion_mask,
            'pre_motion_mask': pre_motion_mask,
            'pre_data': pre_data,                   # not used in dlow/agentformer
            'fut_data': fut_data,                   # not used in dlow/agentformer
            'heading': heading,
            'valid_id': valid_id,
            'traj_scale': self.traj_scale,
            'pred_mask': pred_mask,
            'scene_map': self.geom_scene_map,
            'seq': self.seq_name,
            'frame': frame
        }


    def getScene(self, scene):
        for seq in self.sequence:
            if (seq.seq_name == scene):
                return seq
        raise Exception(f'Scene {scene} not found!')




# ======================================================================
# ------------------------- HELPER FUNCTIONS ---------------------------
# ======================================================================

def dset2scene(dataset):
    if dataset == 'eth':
        test = ['biwi_eth']
    elif dataset == 'hotel':
        test = ['biwi_hotel']
    elif dataset == 'zara1':
        test = ['crowds_zara01']
    elif dataset == 'zara2':
        test = ['crowds_zara02']
    elif dataset == 'univ':
        test = ['students001', 'students003']
    return test[-1]

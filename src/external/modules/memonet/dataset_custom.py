import numpy as np

# Memonet modules
from utils.config import Config
from data.dataloader import data_generator


class Dataset_Custom(data_generator):
    """
    data_generator inherited class that defines the __getitem__ function to allow us
    to index specific members of the dataset. The data is in the form of torch.Tensor.
    """
    def __init__(self, parser, log, split='train', phase='training'):
        super(Dataset_Custom, self).__init__(parser, log, split, phase)
        self.parser = parser
        self.valid_sample_list = self.get_valid_samples()   # consider only the samples that are valid (i.e., have non-null data)


    def __getitem__(self, idx):

        if type(idx)==int:
            sample_idx = self.valid_sample_list[idx]
            seq_idx, frame_idx = self.get_seq_and_frame(sample_idx)
            seq = self.sequence[seq_idx]
            data = seq(frame_idx)
            data['pid_mask'] = None
            return data
            
        elif type(idx)==tuple:
            fid, pid = idx
            # get the scene preprocessor
            test_scene = get_seq_name(self.parser.dataset)
            for preprocesor in self.sequence:
                if preprocesor.seq_name == test_scene:
                    scene = preprocesor
            try:
                data = scene(fid)   # get the data at the corresponding fid and pid
            except:
                raise Exception(f'The given (fid={fid},pid={pid}) pair is not valid!')
            
            # get the fids and pids of all traj in the batch
            data['fids'] = np.array([[fid-(7-i) for i in range(8)] for pid in data['valid_id']])    # (B, 8)
            data['pids'] = np.array(data['valid_id'])   # (B,)
            return data

        """
        data = {
            'pre_motion_3D': pre_motion_3D,         # (N, 8, 2)
            'fut_motion_3D': fut_motion_3D,         # (N, 12, 2)
            'fut_motion_mask': fut_motion_mask,
            'pre_motion_mask': pre_motion_mask,
            'pre_data': pre_data,
            'fut_data': fut_data,
            'heading': heading,
            'valid_id': valid_id,                   # list of valid path_ids at this frame
            'traj_scale': self.traj_scale,
            'pred_mask': pred_mask,
            'scene_map': self.geom_scene_map,
            'seq': self.seq_name,
            'frame': frame                          # frame_id
        }
        """

    def get_fids(self):
        pass


    def get_valid_samples(self):
        # fetch sample idx for all non-null data
        valid_samples = []
        for idx in self.sample_list:
            seq_idx, frame_idx = self.get_seq_and_frame(idx)
            seq = self.sequence[seq_idx]
            data = seq(frame_idx)
            if data is not None:
                valid_samples.append(idx)
        return valid_samples



# ======================================================================
# ------------------------- HELPER FUNCTIONS ---------------------------
# ======================================================================

def get_seq_name(dataset):
    if dataset == 'eth':
        test = ['biwi_eth']
    elif dataset == 'hotel':
        test = ['biwi_hotel']
    elif dataset == 'zara1':
        test = ['crowds_zara01']
    elif dataset == 'zara2':
        test = ['crowds_zara02']
    elif dataset == 'univ':
        test = ['students001', 'students003']   # NOTE: WE HARDCODE TESTING TO ONLY 003 SCENE
    return test[-1]

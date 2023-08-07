import os
import pandas as pd
import numpy as np


class ETH_UCY():
    """
    Class definition for the ETH-UCY dataset
    """
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.df = pd.read_csv(data_map[dataset],
                              sep='\t',
                              index_col=False,
                              header=None,
                              names=['fid', 'pid', 'x', 'y'],
                              dtype={'fid': int, 'pid': int, 'x': float, 'y': float})
        self.H = pd.read_csv(homography_map[dataset],
                             sep='\t',
                             index_col=False,
                             header=None,
                             dtype=float).values
        self.H_inv = np.linalg.inv(self.H)
        if dataset in ['zara1', 'zara2', 'univ']:
            self.H_inv = self.H_inv[[1,0,2]]


    def __len__(self):
        return self.df.shape[0]


    def __getitem__(self, idx):
        fid, pid = idx
        mask = (self.df['fid'] == fid) & (self.df['pid'] == pid)
        return self.df[mask][['x', 'y']].values


    def getValid(self, fid, past_horizon=7, future_horizon=12):
        """
        get all trajectories at the current fid that have a given 
        past traj length and future traj length
        """
        lfid = fid - 10*past_horizon
        ufid = fid + 10*future_horizon
        fids = self.df[(self.df.fid >= lfid) & (self.df.fid <= ufid)].fid.unique()
        pids = set(self.df[self.df.fid==fid].pid)
        for fid in fids:
            pids &= set(self.df[self.df.fid == fid].pid)
        mask = (self.df.fid.isin(fids)) & (self.df.pid.isin(pids))
        return self.df[mask]

    def get(self, ts):
        lfid, ufid = ts
        mask = (self.df.fid >= lfid) & (self.df.fid <= ufid)
        return self.df[mask]


    def world2pixel(self, x):
        """
        transforms the given position (x,y) from world-coordinates (meters) to 
        pixel coordinates (p1,p2) using the homography matrix of the dataset
        x: (*,3) or (*,2)

        https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
        a*(x,y,1) = H @ (p1,p2,1)
        """
        x = np.einsum('i...j->j...i', x)        # (2, *)
        if len(x)==2:
            ones = np.ones_like(x)[:1]
            x = np.concatenate((x, ones), axis=0)
        assert len(x)==3

        pix = self.H_inv @ x                    # (3,*) <-- (3,3) @ (3,*)
        pix = np.einsum('i...j->j...i', pix)    # (*,3)
        pix /= pix[...,2:]                      # scale positions
        return pix[...,:2]                      # (*,2)






DATA_DIR = os.path.dirname(__file__)
data_map = {
    'eth': f'{DATA_DIR}/eth_ucy/annotated/eth/test/biwi_eth.txt',
    'hotel': f'{DATA_DIR}/eth_ucy/annotated/hotel/test/biwi_hotel.txt',
    'zara1': f'{DATA_DIR}/eth_ucy/annotated/zara1/test/crowds_zara01.txt',
    'zara2': f'{DATA_DIR}/eth_ucy/annotated/zara2/test/crowds_zara02.txt',
    'univ': f'{DATA_DIR}/eth_ucy/annotated/univ/test/students003.txt'       # NOTE only use 003
}

homography_map = {
    'eth': f'{DATA_DIR}/eth_ucy/raw/eth/H.txt',
    'hotel': f'{DATA_DIR}/eth_ucy/raw/hotel/H.txt',
    'zara1': f'{DATA_DIR}/eth_ucy/raw/zara1/H.txt',
    'zara2': f'{DATA_DIR}/eth_ucy/raw/zara2/H.txt',
    'univ': f'{DATA_DIR}/eth_ucy/raw/univ/H.txt'
}
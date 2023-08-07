

import math
import os
import cv2
import matplotlib.markers as markers
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D
import numpy as np

from data.ethucy import ETH_UCY


video_paths = {
    'eth': 'data/eth_ucy/raw/eth/seq_eth.avi',
    'hotel': 'data/eth_ucy/raw/hotel/seq_hotel.avi',
    'zara1': 'data/eth_ucy/raw/zara1/crowds_zara01.avi',
    'zara2': 'data/eth_ucy/raw/zara2/crowds_zara02.avi',
    'univ': 'data/eth_ucy/raw/univ/students003.avi',
}




def visualize_agents_raw(dataset, fid, pids, agent_id, coeffs, savefile='.'):
    """
    Plot the trajectories of all given agents. The sensitivity of each agent's position
    along the trajectory is emphasized in the plot. 
    coeffs:     (B*8*2+1) optimal linear coefficients
    fid:        frame id to be visualized
    pids:       (B,) np array of path ids corresponding to each agent trajectory
    agent_id:   pid of the predicted agent
    """

    # get the raw data (in world coordinates)
    ethucy = ETH_UCY(dataset)
    df = ethucy.get([fid-70, fid])

    # get the past trajectories of all agents in pixel coordinates
    trajs = []
    for pid in pids:
        traj = df[df.pid == pid][['x','y']].values
        traj_px = ethucy.world2pixel(traj)
        trajs.append(traj_px)   # (~8, 2)

    # get the future trajectory of the predicted agent
    df = ethucy.get([fid+10, fid+120])
    future = df[df.pid == agent_id][['x','y']].values   # (12, 2)

    # get the scene frame
    frame = extractFrames(video_paths[dataset], [fid], verbose=False)[0]
    b,g,r = cv2.split(frame)
    frame = cv2.merge((r,g,b))

    # plot trajectories for each set of coefficients
    plotTrajectories_raw(frame, trajs, coeffs, pids==agent_id, savefile=savefile)



def plotTrajectories_raw(frame, trajs, coeffs, labels, savefile):
    """
    plots trajectories on the given frame with size depending on the coeff maginitude
    trajs:      [B, ~8, 2]
    coeffs:     (B*8*2 + 1) optimal linear coefficients
    labels:     (B,) one hot encoded array specifying the index of the agent
    """
    bias = coeffs[-1]
    coeffs = np.reshape(coeffs[:-1], (-1, 8, 2))    # (B, 8, 2)
    maxnorm = np.max(np.sqrt(np.sum(coeffs**2, axis=-1)))

    plt.imshow(frame)
    for traj, coeff, label in zip(trajs, coeffs, labels):  # (~8,2), (8,2), bool
        COLOR = 'green' if label else 'tab:blue'
        plt.plot(traj[...,1], traj[...,0], color=COLOR, alpha=0.5, linewidth=2)
        plotMagnitudes(traj[...,1],
                       traj[...,0],
                       coeff,
                       scale=50/maxnorm,
                       color=COLOR)

    #plt.plot(330.23, 232.95, 'og', markersize=10)
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    plt.savefig(savefile)   # NOTE: overwrites the heatmap for each output dim
    plt.show()
    plt.clf()







def visualize_agents(dataset, fid, pids, agent_id, coeffs, radius=0.1, future=None, pred=None, pred_adv=None, plot_adversarial=False, savefile='/images', annotate=True, show=True):
    """
    Plot the trajectories of all given agents. The sensitivity of each agent's position
    along the trajectory is emphasized in the plot. 
    coeffs:     (B*8*2+1) optimal linear coefficients
    fid:        frame id to be visualized
    pids:       (B,) np array of path ids corresponding to each agent trajectory
    agent_id:   pid of the predicted agent
    future:     (12, 2) future traj for the agent
    pred:       (12, 2) predicted traj for the agent
    pred_adv:   (12, 2) predicted adversarial trajectories for the agent
    """ 
    if plot_adversarial:
        assert (radius is not None) and (pred is not None) and (pred_adv is not None)
    coeffs = coeffs[:-1].reshape(-1, 8, 2)  # (B, 8, 2)
    assert len(coeffs)==len(pids)
   
    # get the raw data (in world coordinates)
    ethucy = ETH_UCY(dataset)
    df = ethucy.get([fid-70, fid])

    # get the past trajectories of all agents in pixel coordinates
    past, past_adv = [], []
    for pid, coeff in zip(pids, coeffs):
        past_wd = df[df.pid == pid][['x','y']].values
        past_px = ethucy.world2pixel(past_wd)
        past.append(past_px)            # (~8, 2)
        if plot_adversarial:
            past_adv_wd = attack(past_wd, coeff, radius)
            past_adv_px = ethucy.world2pixel(past_adv_wd)
            past_adv.append(past_adv_px)    # (~8, 2)
        else:
            past_adv.append(None)

    # get the future trajectory of the agent in pixel coordinates
    df = ethucy.get([fid+10, fid+120])
    future_raw = df[df.pid == agent_id][['x','y']].values
    T_world = findTransform(pts_src=future, pts_dst=future_raw, type='homography', method=cv2.RANSAC)
    future = ethucy.world2pixel(future_raw)

    # get the predicted trajectory in pixel coordinates
    if plot_adversarial:
        pred = warpPoints(T_world, pred, type='homography')
        pred_adv = warpPoints(T_world, pred_adv, type='homography')
        pred = ethucy.world2pixel(pred)
        pred_adv = ethucy.world2pixel(pred_adv)


    # get the scene frame
    frame = extractFrames(video_paths[dataset], [fid], verbose=False)[0]
    b,g,r = cv2.split(frame)
    frame = cv2.merge((r,g,b))

    # plot trajectories for each set of coefficients
    plotTrajectories(frame,
                     past,
                     future,
                     pred,
                     past_adv,
                     pred_adv,
                     coeffs,
                     pids,
                     pids==agent_id,
                     plot_adversarial,
                     savefile=savefile,
                     annotate=annotate,
                     show=show)




def plotTrajectories(frame, past, future, pred, past_adv, pred_adv, coeffs, pids, labels, plot_adversarial=False, savefile='/images', annotate=True, show=True):
    """
    plots trajectories on the given frame with size depending on the coeff maginitude
    past:       [B, ~8, 2]
    future:     (12, 2)
    pred:       (12, 2)
    past_adv:   [B, ~8, 2]
    pred_adv:   (12, 2)
    coeffs:     (B, 8, 2) optimal linear coefficients
    pids:       (B,) path ids
    labels:     (B,) one hot encoded array specifying the index of the agent
    """
    maxnorm = np.max(np.sqrt(np.sum(coeffs**2, axis=-1)))

    plt.imshow(frame)
    for traj, traj_adv, coeff, pid, label in zip(past, past_adv, coeffs, pids, labels):  # (~8,2), (~8,2), (8,2), int, bool
        COLOR = 'lime' if label else 'deepskyblue'
        if plot_adversarial:
            COLOR = 'green' if label else 'tab:blue'
        # plot trajectories
        if plot_adversarial:
            plt.plot(traj[...,1], traj[...,0], color=COLOR, alpha=1, linewidth=2)
            plt.plot(traj_adv[...,1], traj_adv[...,0], color='red', alpha=0.5, linewidth=2)
        else:
            plt.plot(traj[...,1], traj[...,0], color=COLOR, alpha=0.3, linewidth=2)        
            plotMagnitudes(traj[...,1],
                           traj[...,0],
                           coeff,
                           scale=70/maxnorm,
                           color=COLOR)
        
        # annotate trajectories
        if traj[-1,1]-traj[-2,1]==0:    # for stability
            traj[-1,1]+=1e-6
        slope = (traj[-1,0]-traj[-2,0])/(traj[-1,1]-traj[-2,1])
        x = traj[-1,1] + 1*(traj[-1,1]-traj[-2,1])
        y = slope*(x - traj[-2,1]) + traj[-2,0]
        if annotate:
            plt.annotate(int(pid),
                        (traj[-1,1], traj[-1,0]),
                        xytext=(x,y),   # textcoords='offset points'
                        ha='center',
                        color=COLOR,
                        path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])


    # plot predicted paths
    if plot_adversarial:
        plt.plot(pred[...,1], pred[...,0], color='orange', linestyle='--', alpha=0.8, linewidth=2)
        plt.plot(future[...,1], future[...,0], color='tab:green', linestyle='--', alpha=1, linewidth=2)
        plt.plot(pred_adv[...,1], pred_adv[...,0], color='red', linestyle='--', alpha=1, linewidth=2)
    else:
        plt.plot(future[...,1], future[...,0], color='lime', linestyle='--', alpha=1, linewidth=2)


    # make legend
    green_line = Line2D([0,1],[0,1], linestyle='-', color='green' if plot_adversarial else 'lime')
    blue_line = Line2D([0,1],[0,1], linestyle='-', color='tab:blue' if plot_adversarial else 'deepskyblue')
    orange_line = Line2D([0,1],[0,1], linestyle='-', color='orange')
    red_line = Line2D([0,1],[0,1], linestyle='-', color='red')
    lines = [blue_line, green_line, orange_line, red_line]
    text = ['neigh. agents', 'true traj.', 'pred. traj.', 'advers. traj.']
    if not plot_adversarial:
        lines = lines[:-2]
        text = text[:-2]
    plt.legend(lines, text)

    # save image
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    plt.savefig(savefile, bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()




def plotMagnitudes(x, y, coeff, scale=1, **kwargs):
    """
    plot the magnitude of coefficients at each point (x,y) as scaled dots
    x:      (<=8,)
    y:      (<=8,)
    coeff:  (8, 2)
    scale:  the scale for the magnitudes
    https://stackoverflow.com/questions/23345565/is-it-possible-to-control-matplotlib-marker-orientation
    """
    sizes = scale * np.sqrt(np.sum(coeff**2, axis=-1))[-len(x):]    # (8,)
    #plt.scatter(x, y, sizes, marker='o', **kwargs)

    for i in range(len(sizes)):
        if i == 0:
            angle = 270 - 180 * math.atan2((y[i+1]-y[i]), (x[i+1]-x[i])) / math.pi
        else:
            angle = 270 - 180 * math.atan2((y[i]-y[i-1]), (x[i]-x[i-1])) / math.pi
        arrow = markers.MarkerStyle(marker='^')#markers.CARETUPBASE)
        arrow._transform = arrow.get_transform().rotate_deg(angle)
        plt.scatter(x[i], y[i], sizes[i], marker=arrow, **kwargs)
    """
    for i in range(len(mag)):
        plt.plot(x[i], y[i], marker='o', markersize=scale*mag[i], **kwargs)
        #plt.scatter(x[i], y[i], marker='o', edgecolors='black', s=scale*mag[i], **kwargs)
    """




# ========================================================================
# =========================== HELPER FUNCTIONS ===========================
# ========================================================================


def extractFrames(vidPath, fids, verbose=False):
    vid = cv2.VideoCapture(vidPath)
    if verbose: print(f'extracting frames {fids} from {vidPath}')
    idx = 0
    frames = []
    notEOF = True
    while notEOF:
        notEOF, image = vid.read()
        if idx in fids:
            frames.append(image)
        idx+=1
    if verbose: print('total number of frames:', idx)
    return frames


def attack(traj, coeffs, radius):
    """
    attacks the given past trajectory to generate adversarial samples
    based on the coefficients and radius
    traj:   (~8, 2) past trajectory
    coeffs: (8, 2)
    """
    sign_grad = np.sign(coeffs[:len(traj)])
    return traj + radius*sign_grad


def findTransform(pts_src, pts_dst, type='affine', **kwargs):
    if type=='affine':
        pts_src = pts_src[:3]
        pts_dst = pts_dst[:3]
        assert len(pts_src) == 3
        assert len(pts_dst) == 3
        H = cv2.getAffineTransform(pts_src, pts_dst, **kwargs)        # (2, 3)
    elif type=='homography':
        assert len(pts_src) >= 4
        assert len(pts_dst) >= 4
        H, mask = cv2.findHomography(pts_src, pts_dst, **kwargs)      # (3, 3)
    return H


def warpPoints(T_mat, pts, type='affine', **kwargs):
    bias = np.ones((pts.shape[0], 1))                   # (12, 1)
    pts_bias = np.concatenate((pts, bias), axis=-1)     # (12, 3)
    pts_bias = np.einsum('i...j->j...i', pts_bias)      # (3, 12)

    if type=='affine':
        assert T_mat.shape == (2,3)
        pts_warp = T_mat @ pts_bias                 # (2, 12)
        return np.einsum('i...j->j...i', pts_warp)  # (12, 2)

    elif type=='homography':
        assert T_mat.shape == (3,3)
        pts_warp = T_mat @ pts_bias                     # (3, 12)
        pts_warp = np.einsum('i...j->j...i', pts_warp)  # (12, 3)
        return (pts_warp/pts_warp[...,2:])[...,:2]      # (12, 2)



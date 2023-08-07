import itertools
import math
import os
import matplotlib.patheffects as PathEffects
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data.ethucy import ETH_UCY

def sensitivity_map(coeffs, lambdas, fids, pids, dataset, savefile, save=True, show=True):
    """
    coefficient analysis of the learned PAC-model
    coeffs:     list of [(B*8*2 + 1) x n] optimal linear coefficients at each output dim
    lambdas:    list of [λ1,...,λn] lambda margins for each output dim
    fids:       (B, 8) np array of frame ids corresponding to each batch trajectory
    pids:       (B,) np array of path ids for each batch trajectory
    """
    ethucy = ETH_UCY(dataset)
    maps = []

    for i in range(len(coeffs)):
        coeff = coeffs[i]   # (B*8*2 + 1)
        eps = lambdas[i]    # λi

        # get PAC-model coefficients
        bias = coeff[-1]
        coeff = np.reshape(coeff[:-1], (-1, 8, 2))    # (B, 8, 2)

        # normalize magnitude of coefficients to 0-1
        coeff = np.abs(coeff)
        max_ = np.max(coeff)
        min_ = np.min(coeff)
        coeff = (coeff - min_)/(max_ - min_)

        # sort by ascending pids
        sort_idx = np.argsort(pids)     # (B,)
        fids = fids[sort_idx]           # (B, 8)
        pids = pids[sort_idx]           # (B,)
        coeff = coeff[sort_idx]         # (B, 8, 2)


        # set up subplots for each trajectory in batch
        fig = plt.figure()
        nrows, ncols = get_shape(len(coeff))
        shape = (nrows, ncols+1)
        i = 0
        for x in range(nrows):
            for y in range(ncols):
                if i >= len(coeff): break
                fid = fids[i]
                pid = pids[i]
                traj = coeff[i]

                # NOTE: trim to only valid positions
                df = ethucy.get([10*fid[-1]-70, 10*fid[-1]])
                ph = len(df[df.pid == pid])
                fid = fid[-ph:]         # (<=8,)
                traj = coeff[i, -ph:]   # (<=8, 2)

                ax = plt.subplot2grid(shape, (x,y))
                im = heatmap(traj,
                            x_labels=['x', 'y'],
                            y_labels=fid,
                            ax=ax,
                            title=f'pid={int(pid)}',
                            # imshow arguments
                            cmap='YlOrRd', #'YlGn',
                            vmin=0,
                            vmax=1)
                
                # display culmulative magnitutde
                if nrows < 3:
                    ax.annotate(f'Σ = \n{np.sum(traj):.2f}',
                                xy=(0.5,0.5),
                                xycoords='axes fraction',
                                horizontalalignment='center',
                                verticalalignment='center')
                else:
                    ax.annotate(f'Σ = \n{np.sum(traj):.2f}',
                                xy=(2,0.5),
                                xycoords='axes fraction',
                                horizontalalignment='center',
                                verticalalignment='center',)
                                #bbox=dict(boxstyle='square,pad=-1', fc='none', ec='none'),
                                #rotation='vertical')
                i+=1

        # create colorbar
        cbar_ax = plt.subplot2grid(shape, (0,ncols), rowspan=nrows)
        cbar_ax.axis('off')
        cbar = colorbar(fig, cbar_ax, im, label=None)

        fig.tight_layout()
        maps.append(fig)
        if save:
            os.makedirs(os.path.dirname(savefile), exist_ok=True)
            plt.savefig(savefile, bbox_inches='tight')   # NOTE: overwrites the heatmap for each output dim
        if show:
            plt.show()
        plt.close(fig)

    return maps



def heatmap(data, x_labels, y_labels, ax=None, title='', **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels
    """
    assert len(x_labels) == data.shape[1]
    assert len(y_labels) == data.shape[0]
    if ax is None:
        ax = plt.gca()
    # plot the heatmap  
    im = ax.imshow(data, **kwargs)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    return im


def colorbar(fig, cax, im, label=None, orientation='left', pad=0):
    """
    Creates a color bar on the specified side of the given axis.
    """
    # create an axis on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes(orientation, size="5%", pad=pad)
    #cbar = fig.colorbar(im, cax=cax)
    cbar = fig.colorbar(im, ax=cax, location='left')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.set_ticks(np.arange(0, 1.1, 0.5))
    cbar.set_ticklabels(['0.0', '0.5', '1.0'])
    cbar.ax.set_ylabel(label, rotation=-90, va="bottom") if label is not None else None
    return cbar

    """
    ax_tr = axes[0,-1]
    ax_br = axes[-1,-1]
    pos_tr = ax_tr.get_position()
    pos_br = ax_br.get_position()
    width = pos_tr.x1 - pos_br.x0
    height = pos_tr.y1 - pos_br.y0

    cbar_ax = fig.add_axes([
        pos_br.x0,  # x0
        pos_br.y0,  # y0
        0.02,       # width
        height      # height
    ])
    cbar = fig.colorbar(im, cax=cbar_ax)
    """



def get_shape(size):
    if size <= 5:
        return (1, size)
    elif size == 9:     # CUSTOM CASES
        return (2, 5)
    else:
        ncols = opt_cols = math.ceil(math.sqrt(size))
        nrows = opt_rows = math.ceil(size/ncols)
        minRemainder = nrows*ncols - (nrows*ncols//size)*size
        while nrows*ncols > size:
            nrows-=1
            ncols+=1
            remainder = nrows*ncols - (nrows*ncols//size)*size
            if remainder < minRemainder:
                opt_cols = ncols
                opt_rows = nrows
        return (opt_rows, opt_cols)







    """
    # NOTE: tutorial on matplotlib heatmaps

    vegetables = ['cucumber', 'tomato', 'lettuce', 'asparagus', 'potato', 'wheat', 'barley']
    farmers = ['Joe', 'Upland', 'Smith', 'Agrifun', 'Organiculture', 'BioGoods Ltd.', 'Cornylee']

    harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                        [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                        [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                        [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                        [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                        [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                        [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
    
    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    #fig.tight_layout()
    plt.show()
    """

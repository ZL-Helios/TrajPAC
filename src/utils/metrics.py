
import numpy as np
from utils.arc import Arc


class MetricFactory:

    def __init__(self, metric):
        metric_dict = {
            'ade': average_displacement_error,
            'l-inf': max_displacement_error,
            'avgLat': lambda y,t : proj_displacement(y, t, metric='avgLat'),
            'avgLong': lambda y,t : proj_displacement(y, t, metric='avgLong'),
            'maxLat': lambda y,t : proj_displacement(y, t, metric='maxLat'),
            'maxLong': lambda y,t : proj_displacement(y, t, metric='maxLong'),
        }
        assert metric in metric_dict
        self.metric = metric_dict[metric]

    def __call__(self, *args, **kwargs):
        return self.metric(*args, **kwargs)


@staticmethod
def squeeze_batches(input):
    """
    squeezes all leading unit dimensions from the given np array
    """
    squeezed_shape = []
    for size in input.shape[:-2]:
        if size != 1:
            squeezed_shape.append(size)
    squeezed_shape += input.shape[-2:]
    return input.reshape(squeezed_shape)


@staticmethod
def average_displacement_error(y, t):
    """
    computes the ADE of (batched) trajectory y wrt the true path t
    y: (*, 12, 2)
    t: (12, 2)
    returns (*,) array of batched ADEs 
    """
    t = squeeze_batches(t)
    #assert t.ndim == 2
    return np.mean(np.sqrt(np.sum((y - t)**2, axis=-1)), axis=-1)


@staticmethod
def max_displacement_error(y, t):
    """
    computes the l-inf norm of (batched) trajectory y wrt the true path t
    y: (*, 12, 2)
    t: (12, 2)
    """
    t = squeeze_batches(t)
    #assert t.ndim == 2
    return np.max(abs(y-t), axis=(-1, -2))


@staticmethod
def proj_displacement(y, t, metric='avgLat'):
    """
    determines the projected displacement between the ground truth and given traj
    y: (*, 12, 2) or (12, 2)
    t: (12, 2)
    returns (*,) or (0,)
    """
    arc = Arc(squeeze_batches(t))
    dev_lat, dev_long = [], []

    if y.ndim == 2:
        dev_lat, dev_long = arc.deviation(y)
    else:
        for traj in y:
            dlat, dlong = arc.deviation(traj)
            dev_lat.append(dlat)
            dev_long.append(dlong)
        dev_lat = np.array(dev_lat)     # (*, 12)
        dev_long = np.array(dev_long)   # (*, 12)
    
    if metric=='avgLat':
        return np.mean(dev_lat, axis=-1)
    elif metric=='avgLong':
        return np.mean(dev_long, axis=-1)
    elif metric=='maxLat':
        return np.max(dev_lat, axis=-1)
    elif metric=='maxLong':
        return np.max(dev_long, axis=-1)
    



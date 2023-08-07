import sys
import time
import argparse
import importlib
import yaml
from easydict import EasyDict as ezdict

import pac_robust
# MIGHT BE BETTER TO JUST CWD
sys.path.append('..')                   # TODO : MAKE SURE THIS RAISES NO CONFLICTS (I THINK IT DOES!)
from utils.auxiliary import Namespace

parser = argparse.ArgumentParser()
# GENERAL
parser.add_argument('-n', '--net', type=str, choices=['PECNet', 'MemoNet', 'Traj++', 'MID'], required=True,
                    help='Which network to be verified')
parser.add_argument('--gpu', '-gpu', action='store_true',
                    help='Set to use GPU (Optional, defualt False)')
# DATASET
parser.add_argument('--dataset', '-d', type=str, choices=['sdd', 'eth', 'hotel', 'univ', 'zara1', 'zara2'], required=True,
                    help='The dataset of the model')
parser.add_argument('--train', '-train', action='store_true',
                    help='Set if you want to verify images in trainset. (optional, only effect on Mnist and Cifar10 models)')
parser.add_argument('--index', '-idx', type=int, default=0,
                    help='The index of the sample for which robustness is verified')
parser.add_argument('--bsize', '-bsize', type=int, default=256,
                    help='The batchsize for the number of trajectories per scene (default=256)')
# PAC MODEL ROBUSTNESS
parser.add_argument('--sigma_robustness', '-sr', type=float, default=0.5,
                    help='The σ-robustness constant used to define robustness of the regression model.')
#parser.add_argument('--radius', '-r', type=float, required=True,
#                    help='The verification radius of the L-inf ball')
parser.add_argument('--epsilon', '-eps', type=float, required=True,
                    help='The error rate of the PAC-model')
parser.add_argument('--eta', '-eta', type=float, required=True,
                    help='The significance level of the PAC-model (1-confidence)')
parser.add_argument('--robust_type', '-rt', type=str, default='pure_robust', choices=['pure_robust', 'label_robust'],
                    help='The type of robustness definition used to evaluate regression models.')
parser.add_argument('--score_fn', '-sfn', type=str, default='ade', choices=['ade', 'l-inf'],
                    help='The main metric for defining the score function of regression models.')
# OPTIMIZATION
parser.add_argument('-solver', '--lpsolver', default='gurobi', choices=['gurobi', 'cbc'],
                    help='The Linear Programming Solver. (Gurobi or CBC, cvxpy default LP solver if not assigned)')
parser.add_argument('-FT', '--FThreshold', type=int, default=2000,
                    help='The sampling threshold for the first focused learning phase. (optional, only effect on Mnist and Cifar10, default 2000)')
parser.add_argument('-ST', '--SThreshold', type=int, default=8000,
                    help='The sampling threshold for the second focused learning phase. (optional, only effect on Mnist and Cifar10, default 8000)')
# BINARY SEARCH
parser.add_argument('--max_radius', '-maxr', type=float, default=2,
                    help='The maximum radius to consider when performing binary search.')
parser.add_argument('--stepsize', '-step', type=float, default=0.1,
                    help='The stepsize to consider when performing binary search.')
args = parser.parse_args()



def find_max_r(args):
    """
    Function that computes the maximum PAC robustness radius of the model.
    A binary traversal is used to efficiently determine the largest radius value.
    """

    def binary_traversal(min_r, max_r, radius_list):
        min_r = round(min_r, 3)     # round to deal with float precision errors
        max_r = round(max_r, 3)
        mid_r = (SCALE*(max_r - min_r)//2)/SCALE + min_r
        mid_r = (100*mid_r//1)/100

        # Base Case
        if mid_r < min_r or mid_r > max_r:
            pass
        # Recursive Case
        else:
            print('\n================ VERIFYING ROBUSTNESS RADIUS {0} ================'.format(mid_r))
            setattr(args['DeepPAC'], 'radius', mid_r)
            isrobust = pac_robust.verify(args)
            if isrobust:
                radius_list.append(mid_r)
                binary_traversal(min_r=mid_r+STEP, max_r=max_r, radius_list=radius_list)
            else:
                binary_traversal(min_r=min_r, max_r=mid_r-STEP, radius_list=radius_list)

    MAX_R = args['DeepPAC'].max_radius
    STEP = args['DeepPAC'].stepsize
    SCALE = 1./STEP

    robust_r = []
    binary_traversal(min_r=0, max_r=MAX_R, radius_list=robust_r)
    print('\nROBUST RADII:', robust_r)
    return robust_r[-1] if len(robust_r) > 0 else 'None'


def get_model_config(model_name):
    cfg_pth = model_config_dict[model_name]
    with open(cfg_pth, 'r') as f:
        config = yaml.safe_load(f)
    return config



model_config_dict = {
    'PECNet': './config/pecnet.yml',
    'MemoNet': './config/memonet.yml',
    'Traj++': './config/trajectron++.yml',
    'MID': './config/mid.yaml'
}

# get parser namespace and model config yaml
args = parser.parse_args()
cfg = get_model_config(args.net)
args_dict = {'DeepPAC' : args, args.net : ezdict(cfg)}

# new namespace with parser arguments and model config yaml 
#vargs = Namespace()
#vargs.update(**vars(args))
#vargs.update(**ezdict(cfg))

start = time.time()
max_radius = find_max_r(args_dict)
print('Max Robustness Radius:', max_radius)
print('Time: ', time.time()-start)

    



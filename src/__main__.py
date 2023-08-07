import time
import yaml
import argparse
from easydict import EasyDict as ezdict
from verify import pac_robust
from utils.auxiliary import Namespace

parser = argparse.ArgumentParser()
# GENERAL
parser.add_argument('-n', '--net', type=str, choices=['PECNet', 'MemoNet', 'Traj++', 'MID', 'MG-GAN', 'AgentFormer'], required=True,
                    help='Which network to be verified')
parser.add_argument('--gpu', '-gpu', action='store_true',
                    help='Set to use GPU (Optional, defualt False)')
parser.add_argument('--save_PAC_model', action='store_true',
                    help='Save the learned linear PAC-model to the models folder')
parser.add_argument('--show_heatmap', action='store_true',
                    help='Show the heatmap after verification')
parser.add_argument('--plot_heatmap', action='store_true',
                    help='Plot the heatmaps of all agents on the frame image')
parser.add_argument('--plot_sens', action='store_true',
                    help='Plot the trajectory sensitivity of all agents on the frame image')
parser.add_argument('--plot_traj', action='store_true',
                    help='Plot the trajectories of all agents on the frame image')
parser.add_argument('--show_plots', action='store_true',
                    help='Display the visualized plots during program execution')
parser.add_argument('--log_ext', type=str, default='results',
                    help='log file extension')
# DATASET
parser.add_argument('--dataset', '-d', type=str, choices=['sdd', 'eth', 'hotel', 'univ', 'zara1', 'zara2'], required=True,
                    help='The dataset of the model')
parser.add_argument('--train', '-train', action='store_true',
                    help='Set if you want to verify images in trainset. (optional, only effect on Mnist and Cifar10 models)')
parser.add_argument('--bsize', '-bsize', type=int, default=256,
                    help='The batchsize for the number of trajectories per scene (default=256)')
parser.add_argument('--index', '-idx', type=int, default=0,
                    help='The index of the sample for which robustness is verified')
parser.add_argument('--fid', '-fid', type=int, default=None,
                    help='The frame id for the current position')
parser.add_argument('--pid', '-pid', type=int, default=None,
                    help='The path id for the current position')
parser.add_argument('--attack_scope', '-as', type=str, default='basic', choices=['basic', 'env', 'full'],
                    help='The type of adversarial situation to verify.'+
                         '\nbasic: analyze robustness to attacks on only the current trajectory'+
                         '\nenv: analyze robustness to environmental attacks'+
                         '\nfull: analyze robustness to both basic and casual attacks')
# PAC MODEL ROBUSTNESS
parser.add_argument('--sigma_robustness', '-sr', type=float, default=1.,
                    help='The σ-robustness constant used to define robustness of the regression model.')
parser.add_argument('--radius', '-r', type=float, required=True,
                    help='The verification radius of the L-inf ball')
parser.add_argument('--epsilon', '-eps', type=float, required=True,
                    help='The error rate of the PAC-model')
parser.add_argument('--eta', '-eta', type=float, required=True,
                    help='The significance level of the PAC-model (1-confidence)')
parser.add_argument('--robust_type', '-rt', type=str, default='pure', choices=['pure', 'label'],
                    help='The type of robustness definition used to evaluate regression models.')
parser.add_argument('--score_fn', '-sfn', type=str, default='ade', choices=['ade', 'l-inf', 'avgLat', 'avgLong', 'maxLat', 'maxLong'],
                    help='The main metric for defining the score function of regression models.')
parser.add_argument('--n_pure_samples', '-nps', type=int, default=0,
                    help='Number of predictions to use for calculating the (minimum) pure robustness.')
# OPTIMIZATION
parser.add_argument('-solver', '--lpsolver', default='gurobi', choices=['gurobi', 'cbc'],
                    help='The Linear Programming Solver. (Gurobi or CBC, cvxpy default LP solver if not assigned)')
parser.add_argument('-FT', '--FThreshold', type=int, default=2000,
                    help='The sampling threshold for the first focused learning phase. (optional, only effect on Mnist and Cifar10, default 2000)')
parser.add_argument('-ST', '--SThreshold', type=int, default=8000,
                    help='The sampling threshold for the second focused learning phase. (optional, only effect on Mnist and Cifar10, default 8000)')
# MISC / ADVERSARIAL ATTACKS
parser.add_argument('--adversary', type=str, default='linear', choices=['linear', 'pgd'],
                    help='The type of adversarial attack performed to find adversaries.')
parser.add_argument('--prediction_type', type=str, default='best_of', choices=['best_of', 'most_likely'],
                    help='Make predictions using either the best-of-20 forcasts or the most likely forcast (mode of latent z)')

model_config_dict = {
    'PECNet': './config/pecnet.yml',
    'MemoNet': './config/memonet.yml',
    'Traj++': './config/trajectron++.yml',
    'MID': './config/mid.yml',
    'MG-GAN': './config/mggan.yml',
    'AgentFormer': './config/agentformer.yml'
}

def get_model_config(model_name):
    cfg_pth = model_config_dict[model_name]
    with open(cfg_pth, 'r') as f:
        config = yaml.safe_load(f)
    return config

# get parser namespace and model config yaml
args = parser.parse_args()
cfg = get_model_config(args.net)

# set index to be a tuple of (fid, pid)
if args.fid and args.pid:
    setattr(args, 'index', (args.fid, args.pid))

args_dict = {
    'DeepPAC' : args,
    args.net : ezdict(cfg)
}


def verify(args):
    if args['DeepPAC'].epsilon > 1. or args['DeepPAC'].epsilon < 0.:
        print('Error: error rate should be in [0,1]')
    if args['DeepPAC'].eta > 1. or args['DeepPAC'].eta < 0.:
        print('Error: significance level should be in [0,1]')
    
    start = time.time()
    pac_robust.verify(args)
    print('\nTime: ', time.time()-start)


if __name__=='__main__':
    verify(args_dict)
    
    

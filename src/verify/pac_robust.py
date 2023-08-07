import math
from multiprocessing import pool
import pickle
from tkinter.tix import Tree
import pandas as pd
import torch
import cvxpy as cp
import numpy as np
import os
import os.path
import importlib
import time
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from utils.gateway import *
from utils.sensitivity import *
from utils.visualize import *
from utils.log import Logger
from utils.arc import Arc
from utils.metrics import MetricFactory

# Global Vars
network = 'Traj++'
dataset = 'eth'
attack_scope = 'basic'
attack_type = 'linear'
index = 0
batchsize = 256

radius = 0.01           # Radius of B(x,r)
error = 0.01            # Correctness: 1-error
significance = 0.01     # Confidence: 1-significance

FThreshold = 20000      # First Try Threshold (multiple of batchsize)
SThreshold = 10000      # Second Try Threshold (multiple of batchsize)
TThreshold = 1600       # Final Samples
n_pure_samples = 1000   # Number of samples to use for minimum pure robustness calcuations
robust_type = 'pure'    # Type of robustness we want to verify
score_fn = 'ade'        # Type of score function to use (ade or fde)
prediction_type = 'best_of'

# constrained optimization params
M = 4096                # input dim
N = 1                   # output dim
MIN = -1000             # min coeff value of affine model
MAX = 1000              # max coeff value of affine model

np.random.seed(0)       # Fix the random seed


max_sample_scores = []  # TODO: DELETE AFTER


def prepare_scenerio_data(args):
    pool_path = (f'pool_data'
                 f'/{network}'
                 f'/{"sdd" if dataset=="sdd" else "eth_ucy"}'
                 f'/{attack_scope}'
                 f'/{prediction_type}'
                 f'/{dataset}_i={index}_r={radius}_FT={FThreshold}_ST={SThreshold}')

    if os.path.exists(pool_path):
        print('\nLoading pooled data...')
        print(pool_path)
        with open(pool_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print('\nPreparing pooled data...')
        data = prepare_pool_data(args)
        os.makedirs(os.path.dirname(pool_path), exist_ok=True)
        with open(pool_path, 'wb') as f:
            pickle.dump(data, f)
    
    # extra sampling if computing *mimimum* pure robustness
    if n_pure_samples > 0:
        extra_pool_path = (f'pool_data'
                        f'/{network}'
                        f'/{"sdd" if dataset=="sdd" else "eth_ucy"}'
                        f'/predictions'
                        f'/{prediction_type}'
                        f'/{dataset}_i={index}_n={n_pure_samples}')

        if os.path.exists(extra_pool_path):
            print('\nLoading extra data...')
            print(extra_pool_path)
            with open(extra_pool_path, 'rb') as f:
                extra = pickle.load(f)
        else:
            print('\nPreparing extra predictions...')
            extra = model_forward(args, x=None, num_samples=n_pure_samples)
            os.makedirs(os.path.dirname(extra_pool_path), exist_ok=True)
            with open(extra_pool_path, 'wb') as f:
                pickle.dump(extra, f)
    else:
        extra = None

    data['center']['pred_batch'] = extra
    return data




def scenario_optimization(data):
    """
    Function for scenario optimization of an affine (𝜂,𝜖)-PAC model in B(x̂,r)
    data:   the pooled data dictionary of numpy arrays
    """
    # define the input x̂ and model output ŷ
    x_hat = data['center']['past']      # (B, 8, 2)
    y_hat = data['center']['pred']      # (1, 12, 2)
    y_label = data['center']['future']  # (1, 12, 2)
    y_extra = data['center']['pred_batch']  # (N, 1, 12, 2)

    # define linear PAC model params
    global M, N
    global max_sample_scores    # TODO: DELETE AFTER
    M = x_hat.size
    N = 1

    # Check the Threshold
    free_v = min(math.floor(SThreshold*error/2-math.log(1/significance)-1), M+1)
    if free_v < 0:
        raise Exception('The Second Threshold for Scenario Optimization is too Small!')

    metric = MetricFactory(score_fn)
    print('\nSetting up scenario optimziation...')
    print('M:', M)
    print('N:', N)
    print('Scene:', dataset)
    print('idx:', index)
    #print('ADE:', np.mean(np.sqrt(np.sum((y_hat - y_label)**2, axis=-1))))
    print(f'{score_fn}:', metric(y_hat.squeeze(0), y_label.squeeze(0)))
    print('Radius:', radius)
    print('Error Ratio:', error)
    print('Significance:', significance)
    print('Free Variables:', free_v)
    print('Robust Type:', robust_type)
    print('Score Method: ', score_fn)


    # Prepare the optimization variable
    var = cp.Variable(M+1)    # optimization variables (i.e., linear weights)
    eps = cp.Variable(1)      # eps is lambda in the paper
    var_opt = []
    eps_opt = []
    margin1, margin2, margin3 = [], [], []

    # COMPONENT LEARNING
    print('\nApplying component-based learning...')
    for i in range(N):
        print('Learning model component {0}:'.format(i))
        # print(f'Learing model component {i}')


        # get input/output for each focused learning phase
        X, score = [], []
        for phase in range(3):
            X_phase = data['noisy'][phase]['past']      # (T, B, 8, 2)
            Y_phase = data['noisy'][phase]['pred']      # (T, 1, 12, 2)
            X.append(flatten(X_phase))                  # (T, B*8*2)

            if robust_type == 'pure':
                y = y_hat       # (1, 12, 2)
                if n_pure_samples > 0:
                    y = get_nearest_neighbor(Y_phase.squeeze(1), y_extra.squeeze(1))
                    y = np.expand_dims(y, axis=1)   # (T, 1, 12, 2)

            elif robust_type == 'label':
                y = y_label     # (1, 12, 2)


            if score_fn == 'ade':
                l2_err = np.sqrt(np.sum((Y_phase - y)**2, axis=-1))         # (T, 1, 12)
                ade = flatten(l2_err).mean(axis=-1, keepdims=True)          # (T, 1)
                score.append(ade[:,i])                                      # only consider component i
                print('max ADE:', np.max(ade))
                print('min ADE:', np.min(ade))
                max_sample_scores.append(np.max(ade))


            elif score_fn == 'l-inf':
                displacement = flatten(abs(Y_phase - y))                    # (T, 1*12*2)
                l_inf = np.amax(displacement, axis=-1, keepdims=True)       # (T, 1)
                score.append(l_inf[:,i])                                    # only consider component i
                print('max displacement:', np.max(l_inf))
                print('min displacement:', np.min(l_inf))

            elif score_fn in ['avgLat', 'avgLong', 'maxLat', 'maxLong']:
                # NOTE: not yet compatible with n_pure_samples > 0 !!!
                arc = Arc(np.squeeze(y, axis=0))
                deviation = []
                for traj in np.squeeze(Y_phase, axis=1):
                    lat_delta, long_delta = arc.deviation(traj)
                    if score_fn == 'avgLat':
                        deviation.append(np.mean(lat_delta))
                    elif score_fn == 'avgLong':
                        deviation.append(np.mean(long_delta))
                    elif score_fn == 'maxLat':
                        deviation.append(np.max(lat_delta))
                    elif score_fn == 'maxLong':
                        deviation.append(np.max(long_delta))
                deviation = np.array(deviation)     # (T,)
                score.append(deviation)
                print('max deviation:', np.max(deviation))
                print('min deviation:', np.min(deviation))


        # First focused learning phase
        print('\tFirst focused learning phase with {0} cases'.format(len(X[0])))
        res = focused_learning(X[0], score[0], var, eps, boost=True, phase=1, verbose=True)
        margin1.append(res['margin'])
        var_save = np.array(var.value)                  # (B*8*2 + 1,)

        # Second focused learning phase
        print('\tSecond focused learning phase with {0} cases'.format(len(X[1])))
        res = focused_learning(X[1], score[1], var, eps, free_v, phase=2, verbose=True)
        margin2.append(res['margin'])
        free_v_ids, fixed_v_ids = res['free_v_ids'], res['fixed_v_ids']
        var_save[free_v_ids] = var.value[free_v_ids]    # only updating the key features

        # Third focused learning phase
        print('\tThird focused learning phase with {0} cases'.format(len(X[2])))
        res = focused_learning(X[2], score[2], var_save, eps, phase=3, verbose=True)
        margin3.append(res['margin'])
        var_opt.append(var_save)
        
        # Use 2nd focused learning phase margin
        eps_opt.append(margin2[-1]) #(min(eps_2, res['margin']))
        print('\tMaximum Margin:', eps_opt[i])

    # output the optimal affine parameters and lambda margin for each output dimension
    return var_opt, margin1, margin2, margin3     # [(B*9*2+1,) x n] and [λ1,...,λn]




def focused_learning(X, score, var, eps, free_v=50, boost=False, phase=1, verbose=False):
    """
    This function applies focused learning for learning a linear (PAC) model
    X:      numpy input of size (T, B*8*2)
    score:  score function of size (T,)
    var:    linear model parameters of size (B*8*2 + 1)
    eps:    PAC model margin (a.k.a. lambda in paper)
    free_v: number of key features to optimize for in the 2nd focused learning phase
    boost:  use regression during first focused learning phase
    phase:  current phase of focused learning (i.e., 1, 2, 3)
    """
    res = {}
    solve_lp = False

    # add bias dimension to input
    bias = np.ones((X.shape[0], 1))
    X = np.append(X, bias, axis=-1)     # (T, B*8*2 + 1)

    if phase == 1:
        if boost:
            # Boosting optimization with least squares
            reg = LinearRegression(fit_intercept=False).fit(X, score)
            var.value = reg.coef_
            delta = X @ var
            res['margin'] = np.max(np.abs(delta.value - score))

        else:
            # Vanilia LP optimization
            delta = X @ var     # (T1,) <- (T1, B*9*2+1) x (B*9*2+1)
            solve_lp = True

    elif phase == 2:
        # Key feature extraction and learning
        var_abs = np.abs(var.value)
        free_v_ids = np.argsort(var_abs)[-free_v:]
        fixed_v_ids = np.argsort(var_abs)[:-free_v]
        free_delta = X[:, free_v_ids] @ var[free_v_ids]         # (T2) << (T2, free) x (free)
        if fixed_v_ids.size > 0:
            # determine the fixed constant
            fixed_const = X[:, fixed_v_ids] @ var[fixed_v_ids]  # (T2) << (T2, fixd) x (fixd)
            const = np.array(fixed_const.value)
        else:
            # there are no fixed variables
            const = 0
        delta = free_delta + const
        res['free_v_ids'] = free_v_ids
        res['fixed_v_ids'] = fixed_v_ids
        solve_lp = True

    elif phase == 3:
        # Determine the max margin
        delta = X @ var     # (T3,) <- (T3, B*9*2+1) x (B*9*2+1)
        res['margin'] = np.max(np.abs(delta - score))


    if solve_lp:
        # Solving the LP problem
        obj = cp.Minimize(eps)
        cons = [
            #var >= MIN,
            #var <= MAX,
            delta - score <= eps,
            score - delta <= eps,
            eps >= 0
        ]
        prob = cp.Problem(obj, cons)
        prob.solve(solver=lp_solver, warm_start=True)
        res['prob.status'] = prob.status    # OPTIMAL, OPTIMAL_INACCURATE, INFEASIBLE, UNBOUNDED,... 
        res['margin'] = prob.value          # λ
        res['var.value'] = var.value        # (B*9*2 + 1,)


    if verbose:
        if type(var) == cp.expressions.variable.Variable:
            delta, var = delta.value, var.value
        l2err = np.mean((delta - score)**2)
        maxeps = np.max(np.abs(delta - score))
        maxcoeff = np.max(np.abs(var))
        print('\t\tAvg L2 Error:', l2err)
        print('\t\tMax Margin:', maxeps)
        print('\t\tMax Delta:', np.max(delta))
        print('\t\tMin Delta:', np.min(delta))

    return res



def verify_PAC_robustness(data, var, eps, sigma, norm='l-inf'):
    """
    Function to verify PAC σ-robustness of a given (𝜂,𝜖)-PAC model in B(x̂,r)
    data:   the center input/output x̂,ŷ data of B(x̂,r) for which robustness is verfified
    var:    [(B*8*2+1,) x n] list of optimal PAC model parameters for each output dim
    eps:    [a1,...,an] list of lambda margins for each output dim of the PAC model
    sigma:  sigma robustness constant
    """
    print('\nVerifying PAC model σ-robustness in B(x,r)...')
    candidate = []
    shape = data['center']['past'].shape        # (B, 8, 2)
    x_hat = data['center']['past'].reshape(-1)  # (B*8*2)

    # verify robustness at B(x̂,r) for each output dimension
    pac_ubound_list = []
    for i in range(N):
        print('Verifying robustness for component {0}:'.format(i))

        # get PAC model parameters
        var_i = var[i]      # (B*8*2 + 1)
        eps_i = eps[i]      # λi
        eps_max = max(eps)  # λmax

        # get the boundary values of B(x̂,r) that produce the max score
        sign_grad = np.sign(var_i[:-1].reshape(shape))              # (B, 8, 2)
        if norm == 'l-inf':
            x_max = x_hat + radius*sign_grad.reshape(-1)            # (B*8*2)
        elif norm == 'l2':
            unit = np.sqrt(np.sum(sign_grad**2))
            x_max = x_hat + radius*(sign_grad/unit).reshape(-1)     # (B*8*2)

        # compute the max score
        bias = np.ones(1)
        x_max = np.append(x_max, bias, axis=-1)     # (B*8*2 + 1)
        score_i = x_max @ var_i                     # (1,)

        # check if PAC upper bound is always less than sigma (σ)
        pac_ubound = score_i + eps_max
        print('\tMaximal PAC Model Value:', score_i)
        print('\tPAC Upper Bound:', pac_ubound)

        # save bounds at dimension
        pac_ubound_list.append(pac_ubound)


        if pac_ubound > sigma:
            print('\tThe PAC model upper bound is greater than σ={0} for component {1}'.format(sigma, i))
            candidate.append(x_max)

    if len(candidate)==0:
        print('Network is PAC-model σ-robust with σ={2} error rate {0} and confidence level {1}'.format(error, 1-significance, sigma))
        return True, pac_ubound_list
    else:
        print('Network is NOT PAC model σ-robust')
        return False, pac_ubound_list



def generate_adversarial_samples(args, data, params=None, attack='linear'):
    """
    Function that generates adversarial samples according to the linear PAC-model or with FGSM
    data:       the center input/output x̂,ŷ data of B(x̂,r) for which robustness is verfified
    params:     [(B*8*2+1,) x n] list of optimal PAC model parameters for each output dim
    """

    x_hat = data['center']['past']          # (B, 8, 2)
    y_hat = data['center']['pred']          # (1, 12, 2)
    y_label = data['center']['future']      # (1, 12, 2)
    y_extra = data['center']['pred_batch']  # (M, 1, 12, 2)
    shape = x_hat.shape

    if attack=='linear':
        assert params is not None
        pred = []
        for i in range(N):
            coeff = params[i]                               # (B*8*2 + 1)
            sign_grad = np.sign(coeff[:-1].reshape(shape))  # (B, 8, 2)
            x_max = x_hat + radius*sign_grad                # (B, 8, 2)
            #print('x_max:', x_max)
            out = model_forward(args, x_max, num_samples=20, verbose=False)
            pred.append(out)
        pred = np.array(pred)   # (N, K=20, 1, 12, 2)

    elif attack=='pgd':
        adversary = model_attack(args, type='pgd')                              # (B, 8, 2)
        pred = model_forward(args, adversary, num_samples=20, verbose=False)    # (K, 1, 12, 2)
        pred = pred[np.newaxis]     # (N=1, K=20, 1, 12, 2)


    if robust_type=='pure':
        y = y_hat   # (1, 12, 2)
        if n_pure_samples > 0:
            y = get_nearest_neighbor(pred.reshape((-1, 12, 2)), y_extra.squeeze(1)) # (N*K, 12, 2)
            y = y.reshape(pred.shape)   # (N, K, 1, 12, 2)

    elif robust_type=='label':
        y = y_label


    if score_fn == 'ade':
        delta = np.sqrt(np.sum((pred - y)**2, axis=-1)) # (N, K, 1, 12)
        score = np.mean(delta, axis=(-1, -2))           # (N, K)
        idx = np.argmax(score, axis=-1)                 # (N,) max ade candidate
        predAdv = pred[np.arange(len(idx)), idx]        # (N, 1, 12, 2)

    elif score_fn == 'l-inf':
        delta = abs(pred - y)                       # (N, K, 1, 12, 2)
        score = np.max(delta, axis=(-1,-2,-3))      # (N, K)
        idx = np.argmax(score, axis=-1)             # (N,) max l-inf candidates
        predAdv = pred[np.arange(len(idx)), idx]    # (N, 1, 12, 2)

    elif score_fn in ['avgLat', 'avgLong', 'maxLat', 'maxLong']:
        arc = Arc(np.squeeze(y, axis=0))
        delta_lat, delta_long = [], []
        for traj_n in pred:
            delta_lat_n, delta_long_n = [], []
            for traj in traj_n:
                dev_lat, dev_long = arc.deviation(traj[0])
                delta_lat_n.append(dev_lat)     # (12,)
                delta_long_n.append(dev_long)   # (12,)
            delta_lat.append(delta_lat_n)       # (K, 12)
            delta_long.append(delta_long_n)     # (K, 12)
        delta_lat = np.array(delta_lat)         # (N, K, 12)
        delta_long = np.array(delta_long)       # (N, K, 12)

        # get maximum deviation prediction
        if score_fn == 'avgLat':
            score = np.mean(delta_lat, axis=-1)     # (N, K)
        elif score_fn == 'avgLong':
            score = np.mean(delta_long, axis=-1)    # (N, K)
        elif score_fn == 'maxLat':
            score = np.max(delta_lat, axis=-1)      # (N, K)
        elif score_fn == 'maxLong':
            score = np.max(delta_long, axis=-1)     # (N, K)
        idx = np.argmax(score, axis=-1)             # (N,) max deviation candidates
        predAdv = pred[np.arange(len(idx)), idx]    # (N, 1, 12, 2)


    # NOTE: added for min pure robustness
    if robust_type=='pure' and n_pure_samples>0:
        y = y[np.arange(len(idx)), idx]     # (N, 1, 12, 2)
        y = y.transpose((1,0,2,3))          # (1, N, 12, 2)


    metric = MetricFactory(score_fn)
    score_vanilla = metric(y_hat.squeeze(0), y_label.squeeze(0))
    score_adv = metric(predAdv.squeeze(1), y_label.squeeze(0))    # (N,)
    score_robust = metric(predAdv.squeeze(1), y.squeeze(0))       # (N,)

    tabular = '\t{name:<20}\t{value:>.8f}'
    print(tabular.format(name=f'vanilla {score_fn}:', value=score_vanilla.max()))
    print(tabular.format(name=f'adversarial {score_fn}:', value=score_adv.max()))
    print(tabular.format(name=f'{robust_type}-robustness {score_fn}:', value=score_robust.max()))

    return predAdv, {
        'vanilla': score_vanilla,
        'adversarial': score_adv,
        'robust': score_robust,
    }


    """
        # NOTE: this is only for score-function = ade
        delta = np.sqrt(np.sum((pred - y)**2, axis=-1))     # (N, K, 1, 12)
        ade = np.mean(delta, axis=(-1, -2))                 # (N, K)
        idx = np.argmax(ade, axis=-1)                       # (N,)
        predAdv = pred[np.arange(len(idx)), idx]            # (N, 1, 12, 2)

        ade_vanilla = np.mean(np.sqrt(np.sum((y_hat - y_label)**2, axis=-1)))
        ade_adv = np.mean(np.sqrt(np.sum((predAdv - y_label)**2, axis=-1)), axis=(1,2)) # (N,)
        ade_robust = np.mean(np.sqrt(np.sum((predAdv - y)**2, axis=-1)), axis=(1,2))    # (N,)
        tabular = '\t{name:<20}\t{value:>.8f}'
        print(tabular.format(name='vanilla ADE:', value=ade_vanilla))
        print(tabular.format(name='adversarial ADE:', value=ade_adv.mean()))
        print(tabular.format(name=f'{robust_type}-robustness ADE:', value=ade_robust.mean()))

        return predAdv, {
            'vanilla': ade_vanilla,
            'adversarial': ade_adv,
            'robust': ade_robust,
        }
    """




def verify(args):
    global radius, error, significance
    global FThreshold, SThreshold, TThreshold
    global device, lp_solver        # device not necessary here?
    global robust_type, score_fn, n_pure_samples
    global network, dataset, index, batchsize
    global attack_scope, attack_type
    global prediction_type
    global max_sample_scores    # TODO: REMOVE AFTER

    # specify network and dataset params
    network = args['DeepPAC'].net
    dataset = args['DeepPAC'].dataset
    attack_scope = args['DeepPAC'].attack_scope
    attack_type = args['DeepPAC'].adversary
    index = args['DeepPAC'].index
    batchsize = args['DeepPAC'].bsize

    # specify (𝜂,𝜖)-PAC model, B(x̂,r) radius, and σ-robustness constants
    radius = args['DeepPAC'].radius
    error = args['DeepPAC'].epsilon
    significance = args['DeepPAC'].eta
    sigma = args['DeepPAC'].sigma_robustness
    robust_type = args['DeepPAC'].robust_type
    score_fn = args['DeepPAC'].score_fn
    n_pure_samples = args['DeepPAC'].n_pure_samples
    prediction_type = args['DeepPAC'].prediction_type

    # specify device and LP solver
    device = 'cuda' if args['DeepPAC'].gpu else 'cpu'
    lp_solver = None
    if args['DeepPAC'].lpsolver == 'gurobi':
        lp_solver = cp.GUROBI
    elif args['DeepPAC'].lpsolver == 'cbc':
        lp_solver = cp.CBC

    # specify focused learning batch sizes
    FThreshold = args['DeepPAC'].FThreshold
    SThreshold = args['DeepPAC'].SThreshold
    TThreshold = math.ceil(2/error*(math.log(1/significance)+1))


    # learn the linear PAC-model and verify the PAC-robustness
    data = prepare_scenerio_data(args)
    start = time.time()
    var_opt, margin1, margin2, margin3 = scenario_optimization(data)    # eps_opt = margin2
    end = time.time()
    isRobust, pac_ubound = verify_PAC_robustness(data, var_opt, margin2, sigma, norm='l-inf')

    delta_t = end - start
    print('\nPAC-Model Learning Time:', delta_t)

    # generate adversarial samples
    pred_adv = None
    ade = {'vanilla': None, 'adversarial': None, 'robust': None}
    if True:
        print(f'Generating {attack_type} adversary')
        pred_adv, ade = generate_adversarial_samples(args, data, var_opt, attack=attack_type)
        pred_adv = pred_adv[0,0]    # (12, 2)


    # save parameters to logger
    log_exp({'λ1': margin1,
            'λ2': margin2,
            'λ3': margin3,
            '|Δ|+λ': pac_ubound,
            'max_sample_score': max_sample_scores[1],
            'δ': ade['vanilla'],
            '*δ': ade['adversarial'],
            '|Δ|': ade['robust']},
            log_ext=args['DeepPAC'].log_ext)

    # visualize data and save to images
    savedir = 'images/{type}/' + f'{dataset}/{network.lower()}/{robust_type}_robustness/'
    savefile = f'{network.lower()}_{dataset}_r={radius}_{index}.png'
    savefile = savedir + savefile


    show_plots = args['DeepPAC'].show_plots
    if args['DeepPAC'].plot_heatmap:
        sensitivity_map(var_opt,
                        margin2,
                        fids=data['fids'],
                        pids=data['pids'],
                        dataset=dataset,
                        savefile=savefile.format(type='heatmap'),
                        show=show_plots)

    if args['DeepPAC'].plot_sens:
        visualize_agents(dataset,
                         fid=index[0],
                         pids=data['pids'],
                         agent_id=index[1],
                         coeffs=var_opt[0],
                         radius=radius,
                         future=data['center']['future'][0],
                         plot_adversarial=False,
                         savefile=savefile.format(type='sensitivity'),
                         annotate=False,
                         show=show_plots)

    if args['DeepPAC'].plot_traj:
        visualize_agents(dataset,
                         fid=index[0],
                         pids=data['pids'],
                         agent_id=index[1],
                         coeffs=var_opt[0],
                         radius=radius,
                         future=data['center']['future'][0],
                         pred=data['center']['pred'][0],
                         pred_adv=pred_adv,
                         plot_adversarial=True,
                         savefile=savefile.format(type='trajectory'),
                         annotate=False,
                         show=show_plots)

    return isRobust




# ========================================================================
# =========================== HELPER FUNCTIONS ===========================
# ========================================================================

def flatten(x):
    return x.reshape(x.shape[0], -1)

def save_pac_model(**kwargs):
    dir = (f'models'
           f'/PAC-Linear-{network}'
           f'/{"sdd" if dataset=="sdd" else "eth_ucy"}'
           f'/{attack_scope}'
           f'/{dataset}_i={index}_r={radius}_FT={FThreshold}_ST={SThreshold}.pkl')
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    with open(dir, 'wb') as f:
        pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)


def log_exp(kwargs, log_ext):
    """
    saves the given kwargs dict into the logger with extension log_ext.
    The following keys must be given:
        λ1...λ3:    margins
        |Δ|+λ:      pac upper bound
        δ:          vanilla ade
        *δ:         adversarial ade
        |Δ|:        robustness ade
    """
    for k in kwargs:
        assert k in ['λ1','λ2','λ3','|Δ|+λ','δ','*δ','|Δ|', 'max_sample_score', 'model_learning_time']

    # save parameters to logger
    exp = {
        'scene': dataset,
        'fid': index[0],
        'pid': index[1],
        'r': radius,
        '𝜖': error,
        '𝜂': significance,
        'forcast_type': prediction_type,
        'robust_type': robust_type,
        'attack_scope': attack_scope,
        'adversary': attack_type,
        'score_fn': score_fn,
        'model': network,
        '1st_phase': FThreshold,
        '2nd_phase': SThreshold,
        '3rd_phase': TThreshold,
    }
    log = pd.DataFrame()
    log = pd.concat([log, pd.DataFrame([exp])], axis=0, ignore_index=True)
    log['component'] = np.arange(N)
    for k,v in kwargs.items():
        log[k] = v
    log.explode(['component','λ1','λ2','λ3','|Δ|+λ','*δ','|Δ|'])   # new row for each list elem

    logger = Logger(dir='logs/', name='log', ext=log_ext, sheet_name=dataset)
    logger.update(log, axis=0, ignore_index=True)
    #print(logger.df)
    logger.save(sheet_name=dataset)


def get_nearest_neighbor(x, y, metric='ade'):
    """
    For each trajectory in batch x, we return the nearest neighbor in y
    based on the given metric
    x:  (T, 12, 2)
    y:  (N, 12, 2)
    """
    x = np.expand_dims(x, axis=1)   # (T, 1, 12, 2)
    delta = x - y                   # (T, N, 12, 2)

    if metric == 'ade':
        l2err = np.sqrt(np.sum(delta**2, axis=-1))  # (T, N, 12)
        ade = np.mean(l2err, axis=-1)               # (T, N)
        idx_min = np.argmin(ade, axis=-1)           # (T,)
        #print(np.max(ade))
        #print(np.min(ade))

    elif metric == 'l-inf':
        linf = np.max(delta, axis=(-1, -2))     # (T, N)
        idx_min = np.argmin(linf, axis=-1)      # (T,)

    return y[idx_min]   # (T, 12, 2)







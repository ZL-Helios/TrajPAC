# NOTE: this file acts as a gateway between DeepPAC and an external project source
import os
import sys
import glob
import re
import functools
import numpy as np
from tqdm import tqdm
from importlib.machinery import SourceFileLoader

from utils.auxiliary import Silence


# scenario dataloader file for each model
dataloader_file = {
    'MemoNet': './external/modules/memonet/scenario.py',
    'Traj++': './external/modules/trajectron/scenario.py',
    'MID': './external/modules/mid/scenario.py',
    'AgentFormer': './external/modules/agentformer/scenario.py',
}
model_file = {
    'AgentFormer': './external/modules/agentformer/model_custom.py',
    'MemoNet': './external/modules/memonet/model_custom.py',
    'Traj++': './external/modules/trajectron/model_custom.py',
    'MID': './external/modules/mid/model_custom.py',
}
model_class = {
    'AgentFormer': 'AgentFormer_Custom',
    'MemoNet': 'MemoNet_Custom',
    'Traj++': 'Trajectron_Custom',
    'MID': 'MID_Custom',
}
for key in dataloader_file:
    dataloader_file[key] = os.path.abspath(dataloader_file[key])
for key in model_file:
    model_file[key] = os.path.abspath(model_file[key])


# ======================================================================
# ----------------------------- DECORATORS -----------------------------
# ======================================================================

def gateway(func):
    """
    decorator that changes to the appropriate working directory & path before call
    and reverts to the original working directory & path after function excecution
    """
    @functools.wraps(func)
    def wrapper_gateway(*args, **kwargs):
        # get current working directory
        CWD = os.getcwd()
        cache = pre_module_cache(CWD)   # clear module cache of local imports

        # add the custom 'external' directory to the sys path
        local_source_dir = os.path.abspath('./external')
        sys.path.insert(0, local_source_dir)

        # change cwd (and sys.path) to be that of the model workspace
        NET = args[0]['DeepPAC'].net 
        MODEL_WORKSPACE = args[0][NET].project_root
        os.chdir(MODEL_WORKSPACE)
        sys.path.insert(0, '.')

        out = func(*args, **kwargs)

        # change cwd back to DeepPAC workspace
        os.chdir(CWD)
        sys.path.remove('.')
        sys.path.remove(local_source_dir)
        post_module_cache(cache)        # add local imports back to module cache
        return out
        
    return wrapper_gateway



def change_directory(dir):

    # Returns a function decorator with argument as the directory to change to
    def decorator_change_directory(func):
        """
        decorator that changes to the appropriate working directory & path before call
        and reverts to the original working directory & path after function excecution
        """
        @functools.wraps(func)
        def wrapper_change_directory(*args, **kwargs):
            # get current working directory
            CWD = os.getcwd()
            cache = pre_module_cache(CWD)   # clear module cache of local imports

            # add the custom 'external' directory to the sys path
            local_source_dir = os.path.abspath('./external')
            sys.path.insert(0, local_source_dir)

            # change cwd (and sys.path) to be the given directory
            os.chdir(dir)
            sys.path.insert(0, '.')

            # CALL FUNCTION
            out = func(*args, **kwargs)

            # change cwd back to DeepPAC workspace
            os.chdir(CWD)
            sys.path.remove('.')
            sys.path.remove(local_source_dir)
            post_module_cache(cache)        # add local imports back to module cache
            return out
        
        return wrapper_change_directory

    return decorator_change_directory


# ======================================================================
# ----------------------------- GATE FUNC. -----------------------------
# ======================================================================

@gateway
def prepare_pool_data(args):
    """
    Function that prepares and saves the pooled data dictionary of the form:
    data:
        center:
            prev:   (B, 8, 2)
            future: (B, 12, 2)
            pred:   (B, 12, 2)
        noisy:
            prev:   [(B, 8, 2)...]
            pred:   [(B, 12, 2)...]
    where key==center indicates the center point of B(x̂,r) and key==noisy indicates
    the noisy phase-focused scenario data.
    """
    # instantiate the Scenario class and obtain the scenario/noisy dataset
    NET = args['DeepPAC'].net
    dataloader_module = load_module(dataloader_file[NET])
    Scenario = getattr(dataloader_module, 'Scenario')
    dataloader = Scenario(args)

    # get pooled data dictonary
    return dataloader[args['DeepPAC'].index]


@gateway
def model_forward(args, x=None, num_samples=1, **kwargs):
    """
    Forward pass of the model at the given scene id OR the input trajectory x (if given).
    args:   experiment/model configurations
    x:      (B, 8, 2) ndarray past traj (optional)
    num_samples: number of samples to output
    """
    # instantiate the model class and obtain the scenario/noisy dataset
    NET = args['DeepPAC'].net
    model_module = load_module(model_file[NET])
    CustomModel = getattr(model_module, model_class[NET])
    model = CustomModel.from_args(args)

    out_batch = []
    for k in tqdm(range(num_samples)):
        with Silence():
            out = model.forcast(args, x, **kwargs)
        out_batch.append(out)
    return np.array(out_batch)  # (K, 1, 12, 2)



@gateway
def model_attack(args, type='pgd', **kwargs):
    """
    adversarial attack of the model at the given (fid,pid)
    args:   experiment/model configurations
    """
    # instantiate the model class and obtain the scenario/noisy dataset
    NET = args['DeepPAC'].net
    model_module = load_module(model_file[NET])
    CustomModel = getattr(model_module, model_class[NET])
    model = CustomModel.from_args(args)
    return model.attack(args, type, **kwargs)





# ======================================================================
# ------------------------- HELPER FUNCTIONS ---------------------------
# ======================================================================

def load_module(module_path):
    # Function that loads the module at a given file path
    head, tail = os.path.split(module_path)
    module_name = tail[:-3]
    return SourceFileLoader(module_name, module_path).load_module()


def pre_module_cache(CWD):
    local_cache = reset_module_cache(CWD)   # save and clear local imports from module cache
    global_cache = set(sys.modules.keys())  # save global imports
    return {
        'local' : local_cache,
        'global' : global_cache,
    }


def post_module_cache(cache):
    # clear module cache of external imports
    for module in set(sys.modules.keys()):
        if module not in cache['global']:
            sys.modules.pop(module)
    # reinitialize module cache with local imports
    load_module_cache(cache['local'])


def reset_module_cache(CWD):
    local_modules = find_local_modules(CWD)
    cache = {}
    for module_name in local_modules:
        module = sys.modules.pop(module_name, False)
        if module:
            cache[module_name] = module
    return cache


def load_module_cache(module_cache):
    for module_name in module_cache:
        sys.modules[module_name] = module_cache[module_name]


def find_local_modules(root):
    """
    Returns a set of all modules within a given root directory.
    i.e., ['utils', 'utils.auxiliary', 'utils.auxiliary.pool', ...]
    """
    res = set()
    queue = [root]
    while queue:
        curDir = queue.pop(0)
        modules = glob.glob(curDir+'/*.py')

        for module in modules:
            rel_module = module[len(root):]
            split_by_module = re.split(r'[/\\.]+', rel_module)[1:-1]  # discard .py extension and leading backslash
            n = len(split_by_module)
            module_names = [     # get list of all hierarchial modules
                '.'.join(split_by_module[:n-idx]).strip('.') 
                for idx in range(n)]
            res.update(module_names)
        
        dirs = glob.glob(curDir+'/*/')
        queue.extend(dirs)
    
    return res



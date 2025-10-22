# TrajPAC: PAC robustness analysis for trajectory forecasting
TrajPAC is a framework for the robustness analysis of trajectory prediction models from a probably approximately correct viewpoint. This repository contains the code for our paper: [TrajPAC: Towards Robustness Verification of Pedestrian Trajectory Prediction Models](https://arxiv.org/abs/2308.05985).

## Installation
### Bare-bones Framework
Our verification method applies to any black-box trajectory prediction model. To install just the basics, we can run the following:
```
conda create --name TrajPAC python=3.10
conda activate TrajPAC
pip install -r requirements.txt
```

### Experiments
To run TrajPAC on the forecasting models analyzed in our paper, we'll need to first install each (pretrained) model separately.

| Model  | Source Code |
| ------------- | ------------- |
| [Trajectron++](https://arxiv.org/abs/2001.03093)  | <https://github.com/StanfordASL/Trajectron-plus-plus>  |
| [MemoNet](https://arxiv.org/abs/2203.11474)  | <https://github.com/MediaBrain-SJTU/MemoNet>  |
| [AgentFormer](https://arxiv.org/abs/2103.14023)  | <https://github.com/Khrylx/AgentFormer>  |
| [MID](https://arxiv.org/abs/2203.13777)  | <https://github.com/Gutianpei/MID>  |

We recommend first cloning the TrajPAC environment using ``$ conda create --name AgentFormer --clone TrajPAC`` and then installing each project's dependencies in the cloned environment. 

### Datasets
We provide the original raw and annotated ETH/UCY dataset [here](https://drive.google.com/drive/folders/1_LTzD3vLVKSBoQ6aWsAcwGgs-HGmIAcM?usp=sharing). The raw data includes both the original recorded scenes as well as their homography mappings. Since most prediction models train on some transformed version of the data, having the raw data allows us to visualize specific instances within each scene across all models.

### Project Structure
Once completed, organize the structure of the workspace as follows:
```
workspace
├── Trajectron++
├── MemoNet-ETH
├── AgentFormer
├── MID
└── TrajPAC 
    └── src
        ├── __main__.py
        ├── requirements.txt
        ├── data
        │   ├── eth_ucy
        │   └── ethucy.py
        ├── images
        ├── logs
        ├── pool_data
        ├── external
        │   └── modules
        │       ├── trajectron
        │       │   ├── scenario.py
        │       │   ├── dataset_custom.py
        │       │   └── model_custom.py
        │       ├── memonet
        │       │   ├── scenario.py
        │       │   ├── dataset_custom.py
        │       │   └── model_custom.py
        │       ├── agentformer
        │       │   ├── scenario.py
        │       │   ├── dataset_custom.py
        │       │   └── model_custom.py
        │       └── mid
        │           ├── scenario.py
        │           ├── dataset_custom.py
        │           └── model_custom.py
        ├── utils
        │   ├── arc.py
        │   ├── auxiliary.py
        │   ├── gateway.py
        │   ├── log.py
        │   ├── metrics.py
        │   ├── sensitivity.py
        │   └── visualize.py
        └── verify
            ├── pac_robust.py
            └── radius.py
```

## Robustness Analysis
To analyze the PAC robustness of a model we can run `__main__.py` in the `TrajPAC/src` directory with key arguments specified by:
* `--net`: which network to verify --Traj++, MemoNet, AgentFormer, MID
* `--dataset`: which scene in the ETH/UCY dataset to verify
* `--fid`: which frame ID of the given scene to analyze
* `--pid`: which path ID of the given frame to analyze
* `--radius`: the defined robustness radius
* `--epsilon`: the error rate of the PAC model
* `--eta`: the significance level of the PAC model
* `--robust_type`: use either the 'pure' or 'label' robustness metric
* `--FThreshold`: number of samples to use in the first focused learning phase
* `--SThreshold`: number of samples to use in the second focused learning phase
* `--adversary`: the adversarial attack method --linear or pgd
* `--log_ext`: extension of the log file to be saved

An example of a fully fleshed out command that performs robustness analysis on Trajectron++ at frame 4400 and path 79 from scene Zara1 is given by
```python . --net Traj++ --gpu --dataset zara1 --radius 0.03 --epsilon 0.01 --eta 0.01 --attack_scope full --adversary linear --robust_type pure --score_fn ade --fid 4400 --pid 79 -FT 30000 -ST 12000 --log_ext analysis --plot_heatmap --plot_sens --plot_traj```

By default, all visualizations are saved in the `TrajPAC/src/images/` folder and all log files containing the optimization details (e.g., PAC upper bounds) are saved in the `TrajPAC/src/logs/` folder.

## Analysis on Custom Models
To perform PAC robustness analysis on your custom models, you need to add black box functions for trajectory predictions at *individual samples* (i.e., for a single fid/pid tuple) into the `TrajPAC/src/external/modules/<custom_model>` directory. The `scenario.py` script is used for generating noisy historical trajectories and saving them as pooled data. The `dataset_custom.py` and `model_custom.py` scripts are for fetching individual fid/pid samples from your dataset and making predictions on those individual samples.

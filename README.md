# TrajRS: Towards Certified Robustness and Defense in Trajectory Prediction

TrajRS is the upgraded v2.0 release of TrajPAC, a trajectory prediction framework which provides a certified robust radius and defense against attacks. Building upon the original PAC-based verification core, TrajRS integrates Randomized Smoothing to provide precise certified robust radii and built-in defense capabilities. This update extends TrajPAC from verification-only analysis to a fully certified, safety-assured tool for trustworthy autonomous driving prediction systems.

## Installation
### Bare-bones Framework
Our framework applies to any black-box trajectory prediction model. To install just the basics, we can run the following:
```
conda create --name TrajRS python=3.10
conda activate TrajRS
pip install -r requirements.txt
```

### Experiments
To run TrajRS on the forecasting models, we'll need to first install each (pretrained) model separately.

| Model  | Source Code |
| ------------- | ------------- |
| [Trajectron++](https://arxiv.org/abs/2001.03093)  | <https://github.com/StanfordASL/Trajectron-plus-plus>  |
| [MemoNet](https://arxiv.org/abs/2203.11474)  | <https://github.com/MediaBrain-SJTU/MemoNet>  |
| [AgentFormer](https://arxiv.org/abs/2103.14023)  | <https://github.com/Khrylx/AgentFormer>  |
| [MID](https://arxiv.org/abs/2203.13777)  | <https://github.com/Gutianpei/MID>  |

We recommend first cloning the TrajRS environment using ``$ conda create --name MID --clone TrajRS`` and then installing each project's dependencies in the cloned environment. 

### Datasets
In the `src/samples/` directory, we have prepared a subset of 500 samples from the ETH/UCY Dataset (`eth.txt`) and another subset of 500 samples from the Stanford Drone Dataset (`sdd.txt`) as examples for testing the smoothed MID model. The complete datasets of the ETH/UCY Dataset and the Stanford Drone Dataset are stored in the `src/data/` directory. To test the certified radius of other samples from the ETH/UCY and Stanford Drone Datasets, simply organize the data in the same format as `eth.txt` and place it in the `src/samples/` directory.

### Project Structure
Once completed, organize the structure of the workspace as follows:
```
workspace
├── Trajectron++
├── MemoNet
├── AgentFormer
├── MID
└── TrajRS 
    └── src
        ├── __main__.py
        ├── requirements.txt
        ├── data
        │   ├── eth_ucy
        │   └── sdd
        ├── samples
        │   ├── eth.txt
        │   └── sdd.txt
        ├── final_result
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
        │   └── metrics.py
        └── verify
            ├── RS_robust.py
            └── radius.py
```

## Robustness Analysis
To analyze the certified robustness radius of a smoothed model we can run `__main__.py` in the `TrajRS/src` directory with key arguments specified by:
* `--files`: which data file to verify (file name of the subset from the dataset of ETH/UCY or SDD in `TrajRS/src/samples/`)
* `--net`: which network to verify --Traj++, MemoNet, AgentFormer, MID
* `--dataset`: which dataset to verify ('eth' for the ETH/UCY dataset and 'sdd' for the Stanford Drone Dataset)
* `--FThreshold`: sampling times n of the TrajRS model
* `--predictnumber`: sampling prediction times n_p for the same sample of the TrajRS model
* `--gaussiansigma`: Gaussian noise parameter sigma of the TrajRS model
* `--alpha`: the significance level of the TrajRS model

An example of a fully fleshed out command that performs certified robustness radii analysis on MID at the ETH/UCY dataset is given by
```python . --files eth --net MID --dataset eth --gpu --FThreshold 10000 --predictnumber 1 --gaussiansigma 0.4 --alpha 0.001```

By default, all evaluation and certified radii results in ADE metric are saved in the `TrajRS/src/final_result/` folder. All evaluation and certified radii results in FDE metric are saved in the `TrajRS/src/fde_final_result/` folder. Predictions of the smoothed model can be found in line "output_clustering" of the csv file. Radii of "robustness for the optimal prediction" can be found in line "r_safe_1" of the csv file and radii of "robustness for all possible predictions" can be found in line "r_safe_N" of the csv file.

## Analysis on Custom Models
To perform TrajRS analysis on your custom models, you need to add black box functions for trajectory predictions at *individual samples* (i.e., for a single fid/pid tuple) into the `TrajRS/src/external/modules/<custom_model>` directory. The `scenario.py` script is used for generating noisy historical trajectories and saving them as pooled data. The `dataset_custom.py` and `model_custom.py` scripts are for fetching individual fid/pid samples from your dataset and making predictions on those individual samples.








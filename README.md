# Reinforcement Learning project

## Software Requirement

- Python 3.9
- Additional python packages are defined in the setup.py
- This document assumes you are running at the top directory

## Directory Organization

```
├── environment.yml                   : Conda setup file with package requirements
├── setup.py                          : Python setup file with requirements files
├── config                	          : folder containing configurations
    └── agent_cfg                     : agent configuration folder
    └── env_cfg                       : env configuration folder
├── drl                	              : folder with drl code
    └── __init__.py                   : make base classes visible
    ├── base         	              : folder containing base classes
        └── __init__.py               : make base classes visible
        └── agent_base.py             : agent base class
    ├── driver                        : folder containing RL steering scripts
        └── driver.py                 : Run scipt
    ├── agents         	              : folder containing agents and registration scripts
        └── __init__.py               : agent registry
        └── registration.py           : script to handle registration
        ├── drl_agents              : folder containing agents
            └── __init__.py           : script to make agents visible
            └── <RLagent>.py          : RL agents

```



## Installing

- Pull code from repo

```
cd drl
```
* Dependencies are managed using the conda environment setup:
```
conda env create -f environment.yml 
conda activate drl_env (required every time you use the package)
```
* Install Data Science Toolkit (via pip):
```
pip install -e . 
```

* Run Code
'''
python drivers/run_opt3.py --agent <agent_name> --env <env_name> --trial 0 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.03 --thresh_quantile 0.95

'''

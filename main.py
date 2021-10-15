"""
To use the module as standalone, reads from config and executes accordingly.
This is a script and only needs to be executed, if the module isn't used as an
imported module.
It serves as guideline, how to use the module in your own pipeline, if imported.
"""

import torch
import yaml

from .ctm.ctm import CTM

# config filename, facilitates usage with different configs
CONFIG_FILENAME = 'config.yml'

# load config
with open(CONFIG_FILENAME, 'r') as cfile:
    cfg = yaml.load(cfile)

#TODO: handle logging
LOG_OPTS = cfg['logging']
LOG_ON = LOG_OPTS['log_on']


# choose device; choose gpu when training/testing on many images, also in the config
device = torch.device('cuda' if torch.cuda.is_available()
        and cfg['device'] == 'gpu' else 'cpu')

#if LOG_ON:
#TODO:LOGGING: log the device state (cpu/gpu)

### DATASET CONFIG ###
dataset_cfg = cfg['dataset']


### MODEL CONFIGS ###
model_cfg = cfg['model']

if model_cfg['parts']['backbone']:
    backbone_cfg = model_cfg['backbone']

    #TODO: init backbone
    backbone_out_dim = 14 #PLACEHOLDER

if model_cfg['parts']['ctm']:
    ctm_cfg = model_cfg['ctm']
    ctm = CTM(ctm_cfg, dataset_cfg, backbone_out_dim)
    #TODO: init ctm

if model_cfg['parts']['metric']:
    metric_cfg = model_cfg['metric']
    #TODO: init metric

### TRAIN/TEST/VAL MODE CONFIG ###
MODE = cfg['mode'] # train, test, val modes
if MODE == 'train':
    train_cfg = cfg['train']




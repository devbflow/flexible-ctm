"""
To use the module as standalone, reads from config and executes accordingly.
This is a script and only needs to be executed, if the module isn't used as an
imported module.
It serves as guideline, how to use the module in your own pipeline, if imported.
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path

from ctm.ctm import CTM
from preprocessing.prepr_model import preprocess_backbone
from metric.metric_module import CosineModule, METRICS

### CONSTANTS ###
# config filename, facilitates usage with different configs
CONFIG_FILENAME = 'config.yml'

## PATH CONSTANTS ##
MODELS_PATH = Path('./models/')
DATA_PATH = Path('./data/')

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
backbone_out_dim = None # same as default value

# in case one disables parts, they will be omitted/be Identity modules
backbone = nn.Identity()
ctm = nn.Identity()
metric = nn.Identity()

## BACKBONE ##
if model_cfg['parts']['backbone']:
    backbone_cfg = model_cfg['backbone']
    model_file = 'backbone_{}.pth'.format(backbone_cfg['name'])
    try:
        # load model
        backbone = torch.load(MODELS_PATH / model_file)
    except FileNotFoundError:
        raise FileNotFoundError("model '{}' does not exist!".format(MODELS_PATH / model_file))

    backbone_out_dim = 14 #PLACEHOLDER specific to used backbone
    backbone = preprocess_backbone(backbone, description=backbone_cfg['name'], dims=backbone_out_dim)

## CTM ##
if model_cfg['parts']['ctm']:
    ctm_cfg = model_cfg['ctm']
    ctm = CTM(ctm_cfg, dataset_cfg, backbone_out_dim)

## METRIC ##
if model_cfg['parts']['metric']:
    metric_cfg = model_cfg['metric']
    # utilize the METRICS dict to get the right metric module
    metric_mod = METRICS[metric_cfg['name']]
    metric = metric_mod(metric_cfg['dim']) # supply the arguments
    #TODO: change the above line

# final model is a Sequential of all prior
model = nn.Sequential(backbone,
                      ctm,
                      metric)

### TRAIN/TEST/VAL MODE CONFIG ###
MODE = cfg['mode'] # train, test, val modes
if MODE == 'train':
    train_cfg = cfg['train']
    batch_size = train_cfg['batch_size']
    epochs = train_cfg['epochs']
    optimizer = train_cfg['optimizer']
elif MODE == 'test':
    #TODO: test mode
    pass
elif MODE == 'val':
    #TODO
    pass
else:
    raise ValueError("Did not recognize mode from config")



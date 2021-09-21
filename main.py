"""
To use the module as standalone, reads from config and executes accordingly.
This is a script and only needs to be executed, if the module isn't used as an
imported module.
It serves as guideline, how to use the module in your own pipeline, if imported.
"""

import torch
import yaml

# load config
with open('config.yml', 'r') as cfile:
    cfg = yaml.load(cfile)

LOG_OPTS = cfg['logging']
LOG_ON = LOG_OPTS['log_on']


# choose device; choose gpu when training/testing on many images, also in the config
device = torch.device('cuda' if torch.cuda.is_available()
        and cfg['hardware'] == 'gpu' else 'cpu')

#if LOG_ON:
#TODO:LOGGING: log the device state (cpu/gpu)


datasets = cfg['datasets']



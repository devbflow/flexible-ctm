"""
To use the module as standalone, reads from config and executes accordingly.
This is a script and only needs to be executed, if the module isn't used as an
imported module.
It serves as guideline, how to use the module in your own pipeline, if imported.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import yaml

from ctm.ctm import CTM
from preprocessing.prepr_model import preprocess_backbone
from metric.metric_module import PairwiseDistModule, METRICS, CosineSimModule
from datasets.loading import *


if __name__ == "__main__":
    ### CONSTANTS ###
    # config filename, facilitates usage with different configs
    CONFIG_FILENAME = 'config.yml'

    ## PATH CONSTANTS ##
    MODELS_PATH = os.path.abspath('./models/')
    DATA_PATH = os.path.abspath('./datasets/')

    ## OTHER CONSTANTS ##
    OPTIMS = {'adam': optim.Adam,
              'sgd': optim.SGD}


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
    dataset_path = os.path.join(DATA_PATH, dataset_cfg['name']) # 

    ### MODEL CONFIGS ###
    model_cfg = cfg['model']
    backbone_outchannels = 0 # same as default value
    backbone_outdim = 0

    # in case one disables parts, they will be omitted/be Identity modules by default
    backbone = nn.Identity()
    ctm = nn.Identity()
    metric = nn.Identity()

    ## BACKBONE ##
    if model_cfg['parts']['backbone']:
        backbone_cfg = model_cfg['backbone']
        model_file = 'backbone_{}.pth'.format(backbone_cfg['name'])
        try:
            # load model
            backbone = torch.load(os.path.join(MODELS_PATH, model_file))
        except FileNotFoundError:
            raise FileNotFoundError("model '{}' does not exist!".format(os.path.join(MODELS_PATH, model_file)))

        backbone_outdim = 14 #PLACEHOLDER specific to used backbone
        backbone, backbone_outshape = preprocess_backbone(backbone, description=backbone_cfg['name'], dims=backbone_outdim)
        backbone_outchannels = backbone_outshape[1]
    ## CTM ##
    if model_cfg['parts']['ctm']:
        ctm_cfg = model_cfg['ctm']
        ctm = CTM(ctm_cfg, dataset_cfg, backbone_outchannels, backbone_outdim)

    ## METRIC ##
    if model_cfg['parts']['metric']:
        metric_cfg = model_cfg['metric']
        # utilize the METRICS dict to get the right metric module
        metric_mod = METRICS[metric_cfg.pop('name')]
        metric = metric_mod(**metric_cfg) # supply the remaining kwargs


    ### TRAIN/TEST MODE ###
    train = cfg['train']
    if train:
        train_cfg = cfg['training']
        epochs = train_cfg['epochs']

        optimizer_cfg = train_cfg['optimizer']
        opt = OPTIMS[optimizer_cfg.pop('name')]
        optimizer = opt(ctm.parameters(), **optimizer_cfg)

        train_loader = get_dataloader(dataset_path=dataset_path,
                                     n_way=dataset_cfg['n_way'],
                                     k_shot=dataset_cfg['k_shot'],
                                     include_query=True,
                                     split='train')

        val_loader = get_dataloader(dataset_path=dataset_path,
                                    n_way=dataset_cfg['n_way'],
                                    k_shot=dataset_cfg['k_shot'],
                                    include_query=True,
                                    split='val')

        ## TRAIN LOOP ##
        for epoch in range(epochs):
            print("Enter epoch loop...")
            for batch, labels in train_loader:
                #print(batch.shape, labels)
                optimizer.zero_grad()
                support_set, query_set, support_labels, query_labels = split_support_query(batch, labels, device=device)
                #print(support_set.shape, support_labels)
                #print(query_set.shape, query_labels)
                # pass through backbone to get feature representation
                supp_features = backbone(support_set).view(-1, 256, 14, 14)
                query_features = backbone(query_set).view(-1, 256, 14, 14)
                #print("after backbone pass")
                #print(supp_features.shape, query_features.shape)
                # pass through CTM to get improved features
                improved_supp, improved_query = ctm(supp_features, query_features)
                #print(improved_supp.shape, improved_query.shape)
                # supply improved features to metric
                metric_score = metric(improved_supp, improved_query, dataset_cfg['n_way'], dataset_cfg['k_shot'])
                break # FIXME
                loss = F.cross_entropy(metric_score, query_labels)
                loss.backward()
                optimizer.step()

            # TODO: validation loop
            with torch.no_grad():
                for support_set, query_set in val_loader:
                    #TODO
                    break
            break
    else: #TEST
        #TODO: test mode
        # simple pass through
        pass

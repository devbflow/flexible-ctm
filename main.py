"""
To use the module as standalone, reads from config and executes accordingly.
This is a script and only needs to be executed, if the module isn't used as an
imported module.
It serves as guideline, how to use the module in your own pipeline, if imported.
"""
from datetime import datetime
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import yaml

from ctm.ctm import CTM
from preprocessing.prepr_model import preprocess_backbone
from metric.metric_module import PairwiseDistModule, METRICS
from datasets.loading import *


if __name__ == "__main__":

    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description="Main script to run training/testing for the CTM based on config file.")
    parser.add_argument('--ctm', metavar='CTM_MODEL', type=str, help="model name for loading a ctm model, necessary for testing")
    parser.add_argument('--metric', metavar='METRIC', type=str, help="metric name for loading a trained metric, should align with the metric in config file")
    parser.add_argument('--cfg', metavar='CFG_PATH', type=str, default='./config.yml', help="optional path to config file (default: './config.yml')")
    parser.add_argument('--models', metavar='MODELS_PATH', type=str, default='./models/', help="optional path to models directory used for loading/saving all models (default: './models/')")
    parser.add_argument('--datasets', metavar='DATA_PATH', type=str, default='./datasets/', help="optional path to datasets (default: './datasets/')")
    args = parser.parse_args()

    ### CONSTANTS ###
    CONFIG_FILE = args.cfg

    ## PATH CONSTANTS ##
    MODELS_PATH = os.path.abspath(args.models)
    DATA_PATH = os.path.abspath(args.datasets)

    ## OTHER CONSTANTS ##
    OPTIMS = {'adam': optim.Adam,
              'sgd': optim.SGD}


    # load config
    with open(CONFIG_FILE, 'r') as cfile:
        print("Loading config file {}".format(CONFIG_FILE))
        cfg = yaml.load(cfile, Loader=yaml.FullLoader)

    ### PREEMPTIVE ERROR CHECKING
    # if test only, raise error as testing untrained models is nonsense
    if cfg['test'] and not cfg['train']:
        if not args.ctm:
            raise RuntimeError("No model being loaded while testing only! Supply '--ctm CTM_MODEL' argument.")
        if not args.metric and cfg['metric']['trainable']:
            raise RuntimeError("No trainable metric being loaded while testing only! Supply '--metric METRIC' argument.")

    # choose device; choose gpu when training/testing on many images, also in the config
    device = torch.device('cuda' if torch.cuda.is_available()
            and cfg['device'] == 'gpu' else 'cpu')
    print("Using device '{}'.".format(device))


    ### DATASET CONFIG ###
    dataset_cfg = cfg['dataset']
    dataset_path = os.path.join(DATA_PATH, dataset_cfg['name']) # 
    print("Using dataset {}.".format(dataset_cfg['name']))


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
            print("Loaded backbone '{}'.".format(model_file))
        except FileNotFoundError:
            raise FileNotFoundError("backbone model '{}' does not exist!".format(os.path.join(MODELS_PATH, model_file)))

        backbone_outdim = 14 #PLACEHOLDER specific to used backbone
        backbone, backbone_outshape = preprocess_backbone(backbone, description=backbone_cfg['name'], dims=backbone_outdim)
        backbone_outchannels = backbone_outshape[1]

    ## CTM ##
    if model_cfg['parts']['ctm']:
        # load previous model if given in args, initialize new ctm model otherwise
        if args.ctm:
            ctm = torch.load(os.path.join(MODELS_PATH, args.ctm))
            print("Loaded CTM module '{}'.".format(args.ctm))
        else:
            ctm_cfg = model_cfg['ctm']
            ctm = CTM(ctm_cfg, dataset_cfg, backbone_outchannels, backbone_outdim)
            print("Initialized new CTM module.")

    ## METRIC ##
    if model_cfg['parts']['metric']:
        metric_cfg = model_cfg['metric']
        if args.metric and metric_cfg['trainable']:
            metric = torch.load(os.path.join(MODELS_PATH, args.metric))
            print("Loaded trainable metric '{}'.".format(args.metric))
        else:
            # utilize the METRICS dict to get the right metric module
            metric_mod = METRICS[metric_cfg.pop('name')]
            metric = metric_mod(**metric_cfg) # supply the remaining kwargs


    ### TRAIN MODE ###
    if cfg['train']:
        train_cfg = cfg['training']
        epochs = train_cfg['epochs']

        optimizer_cfg = train_cfg['optimizer']
        opt = OPTIMS[optimizer_cfg.pop('name')]
        # catch potential NameError if metric is disabled
        try:
            if metric_cfg['trainable']:
                optimizer = opt([ctm.parameters(),
                                 metric.parameters()], **optimizer_cfg)
            else:
                optimizer = opt(ctm.parameters(), **optimizer_cfg)
        except NameError:
            optimizer = opt(ctm.parameters(), **optimizer_cfg)

        train_loader = get_dataloader(dataset_path=dataset_path,
                                     n_way=dataset_cfg['n_way'],
                                     k_shot=dataset_cfg['k_shot'],
                                     include_query=True,
                                     split='train')

        ## TRAIN LOOP ##
        for epoch in range(1, epochs+1):
            epoch_mean_tr_loss = 0 # mean train loss

            # TRAINING #
            ctm.train()
            metric.train()
            for i, (batch, labels) in enumerate(train_loader):
                #if i % 10 == 0:
                #    print("Iteration: {}".format(i+1))
                optimizer.zero_grad()
                # split up batch into support/query sets/labels
                support_set, query_set, support_labels, query_labels = split_support_query(batch, labels, device=device)

                # pass through backbone to get feature representation
                supp_features = backbone(support_set).view(-1, 256, 14, 14)
                query_features = backbone(query_set).view(-1, 256, 14, 14)

                # pass through CTM to get improved features
                improved_supp, improved_query = ctm(supp_features, query_features)

                # supply improved features to metric
                metric_score = metric(improved_supp, improved_query, dataset_cfg['n_way'], dataset_cfg['k_shot'])

                # calculate cross-entropy loss and backprop 
                targets = make_crossentropy_targets(support_labels, query_labels, dataset_cfg['k_shot'])
                loss = F.cross_entropy(metric_score, targets)
                epoch_mean_tr_loss += loss.detach()
                #print(loss)
                loss.backward()
                optimizer.step()
            epoch_mean_tr_loss /= len(train_loader)

            if epoch-1 % 10 == 0:
                print("Epoch {} Mean Train Loss: {}".format(epoch, epoch_mean_tr_loss))

        # save ctm model
        cur_time = datetime.now().isoformat()
        ctm_fname = "ctm_n:{}_k:{}_{}".format(dataset_cfg['n_way'], dataset_cfg['k_shot'], cur_time)
        print("Saving CTM model {}".format(ctm_fname))
        torch.save(ctm, os.path.join(MODELS_PATH, ctm_fname))
        print("Model saved under {}".format(os.path.join(MODELS_PATH, ctm_fname)))

        # save metric if trainable
        try:
            if metric_cfg['trainable']:
                metric_fname = "metric_{}_{}".format(metric_cfg['name'], cur_time)
                print("Saving metric module {}".format(metric_fname))
                torch.save(metric, os.path.join(MODELS_PATH, metric_fname))
                print("Metric saved under {}".format(os.path.join(MODELS_PATH, metric_fname)))
        except NameError:
            # no trainable metric module, no need to save
            pass

    ## TEST MODE ##
    if cfg['test']:

        test_loader = get_dataloader(dataset_path=dataset_path,
                                     n_way=dataset_cfg['n_way'],
                                     k_shot=dataset_cfg['k_shot'],
                                     include_query=True,
                                     split='test')

        mean_acc = 0
        total_corr = 0
        total_num = 0
        ctm.eval()
        metric.eval()
        # TEST LOOP #
        with torch.no_grad():
            for batch, labels in test_loader:
                # split up batch into support/query sets/labels
                support_set, query_set, support_labels, query_labels = split_support_query(batch, labels, device=device)

                # pass through backbone to get feature representation
                supp_features = backbone(support_set).view(-1, 256, 14, 14)
                query_features = backbone(query_set).view(-1, 256, 14, 14)

                # pass through CTM to get improved features
                improved_supp, improved_query = ctm(supp_features, query_features)

                # supply improved features to metric and compare prediction to targets
                metric_score = metric(improved_supp, improved_query, dataset_cfg['n_way'], dataset_cfg['k_shot'])
                targets = make_crossentropy_targets(support_labels, query_labels, dataset_cfg['k_shot'])
                pred = metric_score.argmax(dim=1)
                #print(pred, targets)
                corr_pred = torch.eq(pred, targets).sum()
                #print(corr_pred)
                total_corr += corr_pred
                total_num += targets.shape[0]

        mean_acc = total_corr / total_num
        print("Mean Accuracy of Test set: {}".format(mean_acc))


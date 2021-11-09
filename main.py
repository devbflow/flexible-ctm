"""
To use the module as standalone, reads from config and executes accordingly.
This is a script and only needs to be executed, if the module isn't used as an
imported module.
It serves as guideline, how to use the module in your own pipeline, if imported.
"""
from datetime import datetime

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
        print("Loading config file {}".format(CONFIG_FILENAME))
        cfg = yaml.load(cfile, Loader=yaml.FullLoader)


    # choose device; choose gpu when training/testing on many images, also in the config
    device = torch.device('cuda' if torch.cuda.is_available()
            and cfg['device'] == 'gpu' else 'cpu')
    print("Using device '{}'.".format(device))


    ### DATASET CONFIG ###
    dataset_cfg = cfg['dataset']
    dataset_path = os.path.join(DATA_PATH, dataset_cfg['name']) # 
    print("Loading dataset {}.".format(dataset_cfg['name']))


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
            print("Loaded backbone '{}''.".format(model_file))
        except FileNotFoundError:
            raise FileNotFoundError("model '{}' does not exist!".format(os.path.join(MODELS_PATH, model_file)))

        backbone_outdim = 14 #PLACEHOLDER specific to used backbone
        backbone, backbone_outshape = preprocess_backbone(backbone, description=backbone_cfg['name'], dims=backbone_outdim)
        backbone_outchannels = backbone_outshape[1]

    ## CTM ##
    if model_cfg['parts']['ctm']:
        ctm_cfg = model_cfg['ctm']
        ctm = CTM(ctm_cfg, dataset_cfg, backbone_outchannels, backbone_outdim)
        print("Initialized new CTM module.")
    ## METRIC ##
    if model_cfg['parts']['metric']:
        metric_cfg = model_cfg['metric']
        # utilize the METRICS dict to get the right metric module
        metric_mod = METRICS[metric_cfg.pop('name')]
        metric = metric_mod(**metric_cfg) # supply the remaining kwargs


    ### TRAIN/TEST MODE ###
    if cfg['train']:
        train_cfg = cfg['training']
        epochs = train_cfg['epochs']

        optimizer_cfg = train_cfg['optimizer']
        opt = OPTIMS[optimizer_cfg.pop('name')]
        # catch potential NameError if metric is disabled
        try:
            if metric_cfg['trainable']:
                optimizer = opt([ctm.parameters(),
                                 metric.parameters()])
            else:
                optimizer = opt(ctm.parameters(), **optimizer_cfg)
        except NameError:
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
        for epoch in range(1, epochs+1):
            epoch_mean_tr_loss = 0 # mean train loss

            # TRAINING #
            ctm.train()
            metric.train()
            for i, (batch, labels) in enumerate(train_loader):
                if i % 10 == 0:
                    print("Iteration: {}".format(i+1))
                optimizer.zero_grad()
                #split up batch into support/query sets/labels
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
                print(loss)
                loss.backward()
                optimizer.step()
                break
            #epoch_mean_tr_loss /= len(train_loader)
            epoch_mean_tr_loss /= epoch # FIXME: replace by above without breaks
            if epoch-1 % 10 == 0:
                print("Epoch {} Mean Train Loss: {}".format(epoch, epoch_mean_tr_loss))

            # VALIDATION #
            epoch_mean_acc = 0 # equal to val loss
            ctm.eval()
            metric.eval()
            with torch.no_grad():
                for batch, labels in val_loader:
                    #split up batch into support/query sets/labels
                    support_set, query_set, support_labels, query_labels = split_support_query(batch, labels, device=device)

                    # pass through backbone to get feature representation
                    supp_features = backbone(support_set).view(-1, 256, 14, 14)
                    query_features = backbone(query_set).view(-1, 256, 14, 14)

                    # pass through CTM to get improved features
                    improved_supp, improved_query = ctm(supp_features, query_features)

                    # supply improved features to metric and compare prediction to targets
                    metric_score = metric(improved_supp, improved_query, dataset_cfg['n_way'], dataset_cfg['k_shot'])
                    targets = make_crossentropy_targets(support_labels, query_labels, dataset_cfg['k_shot'])
                    if len(targets.shape) > 1:
                        # both classes are the same
                        continue
                    pred = metric_score.argmax(dim=1)
                    print(pred, targets)
                    corr_pred = torch.eq(pred, targets).sum()
                    print(corr_pred)
                    epoch_mean_acc += corr_pred.detach() / targets.shape[0]
                    break
                #epoch_mean_acc /= len(val_loader)
                epoch_mean_acc /= epoch #FIXME: as above
            print("Epoch {} Mean Val Accuracy: {}".format(epoch, epoch_mean_acc))
            break
        # save ctm model
        '''
        cur_time = datetime.now().isoformat()
        ctm_fname = "ctm_n:{}_k:{}_{}".format(dataset_cfg['n_way'], dataset_cfg['k_shot'], cur_time)
        print("Saving CTM model {}".format(ctm_fname))
        torch.save(ctm, os.path.join(MODELS_PATH, ctm_fname))
        print("Model saved under {}".format(os.path.join(MODELS_PATH, ctm_fname)))
        '''
        # save metric if trainable
    else:
        ## TEST ##
        test_loader = get_dataloader(dataset_path=dataset_path,
                                     n_way=dataset_cfg['n_way'],
                                     k_shot=dataset_cfg['k_shot'],
                                     include_query=True,
                                     split='test')
        ctm.eval()
        metric.eval()
        with torch.no_grad():
            for batch, labels in test_loader:
                
        #TODO: test mode
        # simple pass through
        pass

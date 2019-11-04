# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth,MSML_Loss
from .center_loss import CenterLoss
from .focal_loss import FocalLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    ## add by liu 10-29#################
                    loss_soft = [xent(cls,target) for cls in score]
                    loss_tri = [triplet(f,target)[0] for f in feat]
                    loss_soft = sum(loss_soft)/len(loss_soft)
                    loss_tri = sum(loss_tri)/len(loss_tri)
                    ########################
                    return loss_soft+loss_tri,loss_soft,loss_tri
                    #return xent(score, target) + triplet(feat, target)[0],xent(score, target),triplet(feat, target)[0]  ## modified by liu 0926
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    elif cfg.DATALOADER.SAMPLER == 'focal_triplet':      ## new add by liu
        def loss_func(score,feat,target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                #FocalLoss(gamma=2,alpha=0.25)(score,target)
                focal = FocalLoss(num_classes,alpha=0.25,gamma=2,use_alpha=True)
                ## add by liu 10-29#################
                loss_soft = [focal(cls,target) for cls in score]
                loss_tri = [triplet(f,target)[0] for f in feat]
                loss_soft = sum(loss_soft)/len(loss_soft)
                loss_tri = sum(loss_tri)/len(loss_tri)
                ########################
                return loss_soft+loss_tri,loss_soft,loss_tri
            else:
                return xent(score, target) + triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'focal_msml':      ## new add by liu
        def loss_func(score,feat,target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                #FocalLoss(gamma=2,alpha=0.25)(score,target)
                focal = FocalLoss(num_classes,alpha=0.25,gamma=2,use_alpha=True)
                msml = MSML_Loss(cfg.SOLVER.MARGIN)
                ## add by liu 10-29#################
                loss_soft = [focal(cls,target) for cls in score]
                loss_tri = [msml(f,target)[0] for f in feat]
                loss_soft = sum(loss_soft)/len(loss_soft)
                loss_tri = sum(loss_tri)/len(loss_tri)
                ########################
                return loss_soft+loss_tri,loss_soft,loss_tri
            else:
                return xent(score, target) + triplet(feat, target)[0]
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'focal_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target),\
                        xent(score,target),triplet(feat,target)[0],center_criterion(feat,target)  ### modified by liu
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        elif cfg.MODEL.METRIC_LOSS_TYPE == 'focal_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                focal = FocalLoss(num_classes,alpha=0.25,gamma=2,use_alpha=True)
                ## add by liu 10-29#################
                ## add by liu 10-31
                loss_center = [center_criterion(f,target) for f in feat]
                loss_soft = [focal(cls,target) for cls in score]
                loss_tri = [triplet(f,target)[0] for f in feat]
                loss_soft = sum(loss_soft)/len(loss_soft)
                loss_tri = sum(loss_tri)/len(loss_tri)
                loss_center = sum(loss_center)/len(loss_center)
                ########################
                return loss_soft+loss_tri+cfg.SOLVER.CENTER_LOSS_WEIGHT *loss_center,loss_soft,loss_tri,loss_center

            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func,center_criterion

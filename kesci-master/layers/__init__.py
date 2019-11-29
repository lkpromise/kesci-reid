# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth,TripletLossAll
from .center_loss import CenterLoss
from .focal_loss import FocalLoss
from .mvp_loss import MVPLoss
#from .arc_loss import ArcCos
from .AMSoftmax import AMSoftmaxLoss
from .ranked_loss import RankedLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)
        focal = FocalLoss(num_classes,alpha=0.25,gamma=2,use_alpha=True)  # triplet loss
        am = AMSoftmaxLoss(2048,num_classes)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_all':
        triplet = TripletLossAll(margin=cfg.SOLVER.MARGIN)
        focal = FocalLoss(num_classes,alpha=0.25,gamma=2,use_alpha=True)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'rank':
        rank_loss = RankedLoss(cfg.SOLVER.MARGIN_RANK,cfg.SOLVER.ALPHA,cfg.SOLVER.TVAL)
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
                    loss_soft = xent(score,target)
                    loss_tri = triplet(feat,target)[0]
                    # loss_soft = sum(loss_soft)/len(loss_soft)
                    # loss_tri = sum(loss_tri)/len(loss_tri)
                    ########################
                    return loss_soft+loss_tri,loss_soft,loss_tri
                    #return xent(score, target) + triplet(feat, target)[0],xent(score, target),triplet(feat, target)[0]  ## modified by liu 0926
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    elif cfg.DATALOADER.SAMPLER == 'softmax_rank':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'rank':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    ## add by liu 10-29#################
                    loss_soft = xent(score,target)
                    loss_rank = rank_loss(feat,target)
                    # loss_soft = sum(loss_soft)/len(loss_soft)
                    # loss_tri = sum(loss_tri)/len(loss_tri)
                    ########################
                    return loss_soft+cfg.SOLVER.WEIGHT*loss_rank,loss_soft,loss_rank
                    #return xent(score, target) + triplet(feat, target)[0],xent(score, target),triplet(feat, target)[0]  ## modified by liu 0926
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    elif cfg.DATALOADER.SAMPLER == 'focal_triplet':      ## new add by liu
        def loss_func(score,feat,target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet' or cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_all' :
                #FocalLoss(gamma=2,alpha=0.25)(score,target)
                
                ## add by liu 10-29#################
                loss_soft = focal(score,target)
                loss_tri = triplet(feat,target)[0]
                # loss_soft = sum(loss_soft)/len(loss_soft)
                # loss_tri = sum(loss_tri)/len(loss_tri)
                ########################
                return loss_soft+loss_tri,loss_soft,loss_tri
            else:
                return xent(score, target) + triplet(feat, target)[0]
    # elif cfg.DATALOADER.SAMPLER == 'focal_msml':      ## new add by liu
    #     def loss_func(score,feat,target):
    #         if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
    #             #FocalLoss(gamma=2,alpha=0.25)(score,target)
    #             focal = FocalLoss(num_classes,alpha=0.25,gamma=2,use_alpha=True)
    #             msml = MSML_Loss(cfg.SOLVER.MARGIN)
    #             ## add by liu 10-29#################
    #             loss_soft = [focal(cls,target) for cls in score]
    #             loss_tri = [msml(f,target)[0] for f in feat]
    #             loss_soft = sum(loss_soft)/len(loss_soft)
    #             loss_tri = sum(loss_tri)/len(loss_tri)
    #             ########################
    #             return loss_soft+loss_tri,loss_soft,loss_tri
    #         else:
    #             return xent(score, target) + triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'am_triplet':      ## new add by liu
        def loss_func(score,feat,target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                #FocalLoss(gamma=2,alpha=0.25)(score,target)
                #focal = FocalLoss(num_classes,alpha=0.25,gamma=2,use_alpha=True)
                #arc = ArcCos(2048,num_classes)
                # output = arc(feat,target)
                ## add by liu 10-29#################
                loss_am = am(feat,target)
                loss_tri = triplet(feat,target)[0]
                ########################
                return loss_am+loss_tri,loss_am,loss_tri
            else:
                return xent(score, target) + triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'focal_mvp':      ## new add by liu
        def loss_func(score,feat,target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                #FocalLoss(gamma=2,alpha=0.25)(score,target)
                focal = FocalLoss(num_classes,alpha=0.25,gamma=2,use_alpha=True)
                ## add by liuk 11-11##
                mvp = MVPLoss(margin=200, same_margin = 200.0)
                # mvp_pos =mvp(feat,feat,target,target,mode='pos')
                # mvp_neg = mvp(feat,feat,target,target,mode='neg')
                ## add by liu 10-29#################
                loss_soft = [focal(cls,target) for cls in score]
                loss_pos = [mvp(f,f,target,target,mode='pos') for f in feat]
                loss_neg = [mvp(f,f,target,target,mode='neg') for f in feat]
                #print(loss_pos,loss_neg)
                loss_tri = loss_neg+loss_pos
                #print(loss_tri)
                loss_soft = sum(loss_soft)/len(loss_soft)
                ## 如果有多个特征再进行添加
                loss_tri = sum(loss_tri)/len(loss_tri)
                ########################
                return loss_soft+(1.0 / 20)*loss_tri,loss_soft,loss_tri
            else:
                return xent(score, target) + triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'focal_part':      ## new add by liu
        def loss_func(score,feat,target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                #FocalLoss(gamma=2,alpha=0.25)(score,target)
                focal = FocalLoss(num_classes,alpha=0.25,gamma=2,use_alpha=True)
                ## add by liuk 11-13##
                feat_dim = 2048
                center = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
                # mvp_pos =mvp(feat,feat,target,target,mode='pos')
                # mvp_neg = mvp(feat,feat,target,target,mode='neg')
                ## add by liu 10-29#################
                loss_soft = [focal(cls,target) for cls in score]
                loss_tri = [triplet(f,target)[0] for f in feat]
                loss_center = center(torch.cat(feat,dim=1),target)
                #print(loss_pos,loss_neg)
                #loss_tri = loss_neg+loss_pos
                #print(loss_tri)
                loss_soft = sum(loss_soft)/len(loss_soft)
                ## 如果有多个特征再进行添加
                loss_tri = sum(loss_tri)/len(loss_tri)
                ########################
                return loss_soft+loss_tri+cfg.SOLVER.CENTER_LOSS_WEIGHT*loss_center,loss_soft,loss_tri
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
                ## add by liu 11-19
                loss_center = center_criterion(feat,target)
                loss_soft = focal(score,target)
                loss_tri = triplet(feat,target)[0]
                #loss_tri = [triplet(f,target)[0] for f in feat]
                #loss_soft = sum(loss_soft)/len(loss_soft)
                #loss_tri = sum(loss_tri)/len(loss_tri)
                #loss_center = sum(loss_center)/len(loss_center)
                ########################
                return loss_soft+cfg.SOLVER.CENTER_LOSS_WEIGHT *loss_center,loss_soft,loss_tri,loss_center

            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func,center_criterion

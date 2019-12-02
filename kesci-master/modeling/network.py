import copy
import torch
from torch import nn
import random

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.nonlocal_se import SENet_local
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x
class se_resnext101(nn.Module):
    in_planes = 2048
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(se_resnext101,self).__init__()
        self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
    def forward(self, x):
        x = self.base(x)
        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        ## 训练amsoftmax时做的改变，不使用该loss 时需要改回去   --add by liuk
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softm

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
class ibn_a(nn.Module):
    in_planes = 2048
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(ibn_a,self).__init__()
        self.base = resnet50_ibn_a(last_stride)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.global_reduction = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.global_reduction.apply(weights_init_kaiming)

        # part

        self.part = Bottleneck(2048, 512)
        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_drop = BatchDrop(0.5, 0.5)
        self.part_reduction = nn.Sequential(
            nn.Linear(2048, 1024, True),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.part_reduction.apply(weights_init_kaiming)
        self.part_bn = nn.BatchNorm1d(1024)
        self.part_softmax = nn.Linear(1024, num_classes,bias=False)
        self.part_softmax.apply(weights_init_classifier)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
    def forward(self, x):
        x = self.base(x)
        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        ## 训练amsoftmax时做的改变，不使用该loss 时需要改回去   --add by liuk
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softm

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
class rank(nn.Module):
    in_planes = 2048
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(rank,self).__init__()
        self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.global_reduction = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.global_reduction.apply(weights_init_kaiming)

        # part

        self.part = Bottleneck(2048, 512)
        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_drop = BatchDrop(0.5, 0.5)
        self.part_reduction = nn.Sequential(
            nn.Linear(2048, 1024, True),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.part_reduction.apply(weights_init_kaiming)
        self.part_bn = nn.BatchNorm1d(1024)
        self.part_softmax = nn.Linear(1024, num_classes,bias=False)
        self.part_softmax.apply(weights_init_classifier)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
    def forward(self, x):
        x = self.base(x)
        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        ## 训练amsoftmax时做的改变，不使用该loss 时需要改回去   --add by liuk
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softm

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
class mix_module(nn.Module):
    def __init__(self,num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        #model_path1 = "/home/liuk/kesci_result/baseline_kesci_focal/test/se_resnext101_model_120.pth"
        #model_path2 = "/home/liuk/kesci_result/ibn_kesci/resnet50_ibn_a_model_120.pth"
        super(mix_module,self).__init__()
        # 修改为预训练模型文件的地址
        model_path1 = "/home/liuk/code/pretrained/se_resnext101_32x4d-3b2fe3d8.pth"
        model_path2 = "/home/liuk/.torch/models/r50_ibn_a.pth"
        self.part1 = se_resnext101(num_classes, last_stride, model_path1, neck, neck_feat, model_name, pretrain_choice)
        self.part2 = ibn_a(2772, last_stride, model_path2, neck, neck_feat, model_name, pretrain_choice)
        # self.part3 = rank(num_classes, last_stride, model_path1, neck, neck_feat, model_name, pretrain_choice)
    def forward(self,x):
        feature1 = self.part1(x)
        feature2 = self.part2(x)
        # feature3 = self.part3(x)

        return torch.cat([feature1,feature2],dim=1)


    
    

        
# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import torch
from torch import nn

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

class PartAttention(nn.Module):

    def __init__(self,num_classes):
        super(PartAttention,self).__init__()
        self.pool2d = nn.AdaptiveMaxPool2d(1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = F.softmax
        self.classifier_part1 = nn.Linear(512,num_classes,bias=False)
        self.classifier_part2 = nn.Linear(512,num_classes,bias=False)
        self.classifier_part3 = nn.Linear(512,num_classes,bias=False)

        self.classifier_part1.apply(weights_init_classifier)
        self.classifier_part2.apply(weights_init_classifier)
        self.classifier_part3.apply(weights_init_classifier)

        self.reduction_1 = nn.Sequential(nn.Conv2d(2048,512,1,bias=False),nn.BatchNorm2d(512),nn.ReLU())
        self.reduction_2 = nn.Sequential(nn.Conv2d(2048,512,1,bias=False),nn.BatchNorm2d(512),nn.ReLU())
        self.reduction_3 = nn.Sequential(nn.Conv2d(2048,512,1,bias=False),nn.BatchNorm2d(512),nn.ReLU())

        self.reduction_1.apply(weights_init_kaiming)
        self.reduction_2.apply(weights_init_kaiming)
        self.reduction_3.apply(weights_init_kaiming)



    def forward(self,x):
         part_1_s = x[:,:,0:5,:]
         part_2_s = x[:,:,5:10,:]
         part_3_s = x[:,:,10:,:]

         part_1 = self.pool2d(part_1_s)
         part_1 = part_1.view(part_1.shape[0],-1)
         part_1 = self.softmax(part_1,dim=1)

         part_2 = self.pool2d(part_2_s)
         part_2 = part_2.view(part_2.shape[0],-1)
         part_2 = self.softmax(part_2,dim=1)

         part_3 = self.pool2d(part_3_s)
         part_3 = part_3.view(part_3.shape[0],-1)
         part_3 = self.softmax(part_3,dim=1)

         part_1_t = self.reduction_1(self.pool(part_1_s)*part_1.unsqueeze(dim=2).unsqueeze(dim=3)).squeeze(dim=3).squeeze(dim=2)
         part_2_t = self.reduction_2(self.pool(part_2_s)*part_2.unsqueeze(dim=2).unsqueeze(dim=3)).squeeze(dim=3).squeeze(dim=2)
         part_3_t = self.reduction_3(self.pool(part_3_s)*part_3.unsqueeze(dim=2).unsqueeze(dim=3)).squeeze(dim=3).squeeze(dim=2)

         #print("it's right")

         cls_1 = self.classifier_part1(part_1_t)
         cls_2 = self.classifier_part2(part_2_t)
         cls_3 = self.classifier_part3(part_3_t)

         return part_1_t,part_2_t,part_3_t,cls_1,cls_2,cls_3
class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            #self.in_planes = 1024
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
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101_local':
            self.base = SENet_local(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)

        # self.pool2d = nn.MaxPool2d(kernel_size=(8,8))
        # self.bottle = Bottleneck(2048,512)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        #self.reduction_global = nn.Sequential(nn.Conv2d(2048,1024, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU())
        # self.reduction_part_1 = nn.Sequential(nn.Conv2d(2048,512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU())
        # self.reduction_part_2 = nn.Sequential(nn.Conv2d(2048,512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU())
        #self.reduction_global.apply(weights_init_kaiming)
        # self.reduction_part_1.apply(weights_init_kaiming)
        # self.reduction_part_2.apply(weights_init_kaiming)
        self.num_classes = num_classes
        #self.attention = PartAttention(self.num_classes)
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
        # self.part_1_classifier = nn.Linear(512, self.num_classes, bias=False)
        # self.part_2_classifier = nn.Linear(512, self.num_classes, bias=False)
        # self.part_1_classifier.apply(weights_init_classifier)
        # self.part_2_classifier.apply(weights_init_classifier)

    def forward(self, x):

        x = self.base(x)

        #part_1_t,part_2_t,part_3_t,cls_1,cls_2,cls_3 = self.attention(x)
        #print(x.shape)

        #global_feat = self.reduction_global(x)

        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        #global_feat = self.reduction_global(global_feat)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # print(global_feat.shape)

        # part = self.bottle(x)
        #
        # part_feature = self.pool2d(x)
        #
        # part_1 = part_feature[:, :, 0:1, :]
        # part_2 = part_feature[:, :, 1:2, :]
        #
        # cls_part1 = self.reduction_part_1(part_1).squeeze(dim=3).squeeze(dim=2)
        # cls_part2 = self.reduction_part_2(part_2).squeeze(dim=3).squeeze(dim=2)
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        #res_feat = torch.cat([global_feat,part_1_t,part_2_t,part_3_t],dim=1)

        if self.training:
            cls_score = self.classifier(feat)
            # cls_score_1 = self.part_1_classifier(cls_part1)
            # cls_score_2 = self.part_2_classifier(cls_part2)
            #return [cls_score,cls_score_1,cls_score_2], [global_feat]
            return [cls_score], [global_feat]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                #global_feat = torch.cat([global_feat,cls_part1,cls_part2],dim=1)
                return global_feat
                # return res_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
class Kesci(nn.Module):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Kesci, self).__init__()
        resnext = SENet(block=SEResNeXtBottleneck,
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
            resnext.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        self.backone = nn.Sequential(
            resnext.layer0,
            resnext.layer1,
            resnext.layer2,
            resnext.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnext.layer3[1:])

        res_g_conv5 = resnext.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        # res_p_conv5.load_state_dict(resnext.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        pool2d = nn.AvgPool2d

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 8))
        self.maxpool_zg_p2 = pool2d(kernel_size=(16, 8))

        self.maxpool_zp2 = pool2d(kernel_size=(8, 8))

        reduction = nn.Sequential(nn.Conv2d(2048,512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)

        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(512, num_classes)
        self.fc_id_2048_1 = nn.Linear(512, num_classes)

        self.fc_id_256_1_0 = nn.Linear(512, num_classes)
        self.fc_id_256_1_1 = nn.Linear(512, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        # print(zg_p1.shape)

        zp2 = self.maxpool_zp2(p2)
        # print(zp2.shape)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]


        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        # print(fg_p1.shape)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_2(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_3(z1_p2).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)

        predict = torch.cat([fg_p1, fg_p2,f0_p2,f1_p2], dim=1)
        # print("suc")
        if self.training:
            return [l_p1,l_p2,l0_p2,l1_p2],[fg_p1,fg_p2] # global feature for triplet loss
        else:
            return predict


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

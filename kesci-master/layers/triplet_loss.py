# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
from torch.autograd import Variable


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            ## 模拟那个max的过程
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            ## 因为marginloss的形式是-y*(x_1-x_2),因此需要把y都设为1同时，an与ap写反。
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an
class TripletLossAll(nn.Module):
    '''
    Original margin ranking loss:
        loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
    
    Let z = -y * (x1 - x2)
    Soft_margin mode:
        loss(x1, x2, y) = log(1 + exp(z))
    Batch_hard mode:
        z = -y * (x1' - x2'),
        where x1' is the max x1 within a batch,
        x2' is the min x2 within a batch
    '''
    def __init__(self, margin=0, batch_hard=False):
        """
        Args:
            margin: int or 'soft'
            batch_hard: whether to use batch_hard loss
        """
        super(TripletLossAll, self).__init__()
        self.batch_hard = batch_hard
        if isinstance(margin, float) or margin == 'soft':
            self.margin = margin
        else:
            raise NotImplementedError(
                'The margin {} is not recognized in TripletLoss()'.format(margin))

    def forward(self, feat, id=None, pos_mask=None, neg_mask=None, mode='id'):
        dist = self.cdist(feat, feat)
        if mode == 'id':
            if id is None:
                 raise RuntimeError('foward is in id mode, please input id!')
            else:
                 identity_mask = Variable(torch.eye(feat.size(0)).byte())
                 identity_mask = identity_mask.cuda() if id.is_cuda else identity_mask
                 same_id_mask = torch.eq(id.unsqueeze(1), id.unsqueeze(0))
                 negative_mask = same_id_mask ^ 1
                 positive_mask = same_id_mask ^ identity_mask
        elif mode == 'mask':
            if pos_mask is None or neg_mask is None:
                 raise RuntimeError('foward is in mask mode, please input pos_mask & neg_mask!')
            else:
                 positive_mask = pos_mask
                 same_id_mask = neg_mask ^ 1
        else:
            raise ValueError('unrecognized mode')
        if self.batch_hard:
            max_positive = (dist * positive_mask.float()).max(1)[0]
            min_negative = (dist + 1e5*same_id_mask.float()).min(1)[0]
            z = max_positive - min_negative
        else:
            pos = positive_mask.topk(k=1, dim=1)[1].view(-1,1)
            positive = torch.gather(dist, dim=1, index=pos)
            pos = negative_mask.topk(k=1, dim=1)[1].view(-1,1)
            negative = torch.gather(dist, dim=1, index=pos)
            z = positive - negative
        if isinstance(self.margin, float):
            b_loss = torch.clamp(z + self.margin, min=0)
        elif self.margin == 'soft':
            b_loss = torch.log(1 + torch.exp(z))
        else:
            raise NotImplementedError("How do you even get here!")
        # index= torch.argmax(b_loss)
        # print(index)
        loss = b_loss.sum()
        return loss,z
            
    def cdist(self, a, b):
        '''
        Returns euclidean distance between a and b
        
        Args:
             a (2D Tensor): A batch of vectors shaped (B1, D)
             b (2D Tensor): A batch of vectors shaped (B2, D)
        Returns:
             A matrix of all pairwise distance between all vectors in a and b,
             will be shape of (B1, B2)
        '''
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return ((diff**2).sum(2)+1e-12).sqrt()

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
if __name__ == '__main__':
    tri = MSML_Loss(0.3)
    global_feat = torch.FloatTensor([[1,2,3,4],[2,3,4,5],[0,2,1,3],[1,3,2,4],[0,0,0,0],[1,2,1,1],[2,3,2,2],[24,4,4,4]])
    labels = torch.LongTensor([1,0,1,0,1,1,0,0])
    loss = tri(global_feat,labels)[0]
    print(loss)

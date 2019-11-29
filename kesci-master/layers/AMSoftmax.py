import torch
import torch.nn as nn
import torch.nn.functional as F

class AMSoftmaxLoss(nn.Module):
  def __init__(self, feat_dim, class_num, 
    margin=0.35, 
    scale=30, 
    gamma=0
  ):
    super(AMSoftmaxLoss, self).__init__()
    self.weight = nn.Parameter(torch.randn(class_num, feat_dim, dtype=torch.float32))
    self.weight.data.uniform_(-1, 1)
    self.margin = margin
    self.scale = scale
    self.gamma = gamma

  def forward(self, x, y):
    # x => (Batch_Size, Feat_Dim), w => (Class_Num, Feat_Dim)
    self.weight.data = F.normalize(self.weight.data, p=2, dim=1)
    x = F.normalize(x, p=2, dim=1)
    cos = torch.mm(x, self.weight.t().cuda())
    one_hot = torch.zeros_like(cos,dtype=torch.uint8)
    one_hot.scatter_(1, y.view(-1, 1), 1)
    softmax_score = self.scale * (cos - self.margin * one_hot.float())

    log_prob = F.log_softmax(softmax_score, dim=-1)[one_hot]
    prob = log_prob.detach().exp()
    loss = -1 * (1 - prob)**self.gamma * log_prob
    loss = loss.mean()
    
    return loss
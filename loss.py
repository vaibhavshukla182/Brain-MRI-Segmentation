import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

def cross_entropy2d(input, target, weight=None, size_average=True):
    
    n, c, h, w = input.size()
    nt, ct, ht, wt = target.size()
    '''
    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt: # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")
    '''
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.contiguous().view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum().float()
    return loss

def loss_ce_t(input,target):
    #input=F.sigmoid(input)
    target_bin=Variable(torch.zeros(1,11,target.shape[2],target.shape[3]).cuda().scatter_(1,target,1))
    return F.binary_cross_entropy_with_logits(input,target_bin)

def dice_loss(input, target):
    target_bin=Variable(torch.zeros(target.shape[0],11,target.shape[2],target.shape[3]).cuda().scatter_(1,target,1))
    smooth = 1.
    iflat = input.view(-1)
    tflat = target_bin.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))

def weighted_loss(input,target,weight,size_average=True):
    n,c,h,w=input.size()
    target_bin=Variable(torch.zeros(n,c,h,w).cuda()).scatter_(1,target,1)
    target_bin=target_bin.transpose(1,2).transpose(2,3).contiguous().view(n*h*w,c).float()
    
    # NHWC
    input=F.softmax(input,dim=1).transpose(1,2).transpose(2,3).contiguous().view(n*h*w,c)
    input=input[target_bin>=0]
    input=input.view(-1,c)
    weight=weight.transpose(1,2).transpose(2,3).contiguous()
    weight=weight.view(n*h*w,1).repeat(1,c)
    '''
    mask=target>=0
    target=target[mask]
    target_bin=np.zeros((n*h*w,c),np.float)
    for i,term in enumerate(target):
        target_bin[i,int(term)]=1
    target_bin=torch.from_numpy(target_bin).float()
    target_bin=Variable(target_bin.cuda())
    '''
    loss=F.binary_cross_entropy(input,target_bin,weight=weight,size_average=False)
    if size_average:
        loss/=(target_bin>=0).data.sum().float()/c
    return loss

def bce2d_hed(input, target):
    """
    Binary Cross Entropy 2-Dimension loss
    """
    n, c, h, w = input.size()
    # assert(max(target) == 1)
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float().cuda()
    target_trans = target_t.clone()
    pos_index = (target_t >0)
    neg_index = (target_t ==0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num*1.0 / sum_num
    weight[neg_index] = pos_num*1.0 / sum_num

    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)
    return loss

def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250,
                          reduce=False, size_average=False)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           size_average=size_average)
    return loss / float(batch_size)

# another implimentation for dice loss
import torch
from torch.autograd import Function, Variable
class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        self.inter = torch.dot(input.view(-1), target.view(-1)) + 0.0001
        self.union = torch.sum(input) + torch.sum(target) + 0.0001
        t = 2 * self.inter.float() / self.union.float()
        return t
    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None
        return grad_input, grad_target
def dice_coeff(input, target):
    target_bin=Variable(torch.zeros(1,11,target.shape[2],target.shape[3]).cuda().scatter_(1,target,1).float())
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(input, target_bin)):
        s = s + DiceCoeff().forward(c[0], c[1])
    return s / (i + 1)


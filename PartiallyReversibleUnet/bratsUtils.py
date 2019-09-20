import medpy.metric.binary as medpyMetrics
import numpy as np
import math
import torch


def softDice(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    if nonSquared:
        union = (pred).sum() + (target).sum()
    else:
        union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + smoothing) / (union + smoothing)

    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])

    return dice.mean()

def dice(pred, target):
    predBin = (pred > 0.5).float()
    return softDice(predBin, target, 0, True).item()

def diceLoss(pred, target, nonSquared=False):
    return 1 - softDice(pred, target, nonSquared=nonSquared)

def bratsDiceLoss(outputs, labels, nonSquared=False):

    #bring outputs into correct shape
    wt, tc, et = outputs.chunk(3, dim=1)
    s = wt.shape
    wt = wt.view(s[0], s[2], s[3], s[4])
    tc = tc.view(s[0], s[2], s[3], s[4])
    et = et.view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask, tcMask, etMask = labels.chunk(3, dim=1)
    s = wtMask.shape
    wtMask = wtMask.view(s[0], s[2], s[3], s[4])
    tcMask = tcMask.view(s[0], s[2], s[3], s[4])
    etMask = etMask.view(s[0], s[2], s[3], s[4])

    #calculate losses
    wtLoss = diceLoss(wt, wtMask, nonSquared=nonSquared)
    tcLoss = diceLoss(tc, tcMask, nonSquared=nonSquared)
    etLoss = diceLoss(et, etMask, nonSquared=nonSquared)
    return (wtLoss + tcLoss + etLoss) / 5

# def weightedDiceLoss(pred, target, smoothing=1, mean = 0.01):
#
#     mean = mean
#     w_1 = 1 / mean ** 2
#     w_0 = 1 / (1 - mean) ** 2
#
#     pred_1 = pred
#     target_1 = target
#     pred_0 = 1 - pred
#     target_0 = 1 - target
#
#     intersection_1 = (pred_1 * target_1).sum(dim=(1, 2, 3))
#     intersection_0 = (pred_0 * target_0).sum(dim=(1, 2, 3))
#     intersection = w_0 * intersection_0 + w_1 * intersection_1
#
#     union_1 = (pred_1).sum() + (target_1).sum()
#     union_0 = (pred_0).sum() + (target_0).sum()
#     union = w_0 * union_0 + w_1 * union_1
#
#     dice = (2 * intersection + smoothing) / (union + smoothing)
#     #fix nans
#     dice[dice != dice] = dice.new_tensor([1.0])
#     return 1 - dice.mean()


def weightedDiceLoss(pred, target, smoothing=1):

    total_pixels_numbers = target.shape[1] * target.shape[2] * target.shape[3]
    target_pixels_numbers = target.sum()
    mean = target_pixels_numbers / total_pixels_numbers
    mean = torch.clamp(mean, min=0.005)

    w_1 = 1 / mean ** 2
    w_0 = 1 / (1 - mean) ** 2

    pred_1 = pred
    target_1 = target
    pred_0 = 1 - pred
    target_0 = 1 - target

    intersection_1 = (pred_1 * target_1).sum(dim=(1, 2, 3))
    intersection_0 = (pred_0 * target_0).sum(dim=(1, 2, 3))
    intersection = w_0 * intersection_0 + w_1 * intersection_1

    union_1 = (pred_1).sum() + (target_1).sum()
    union_0 = (pred_0).sum() + (target_0).sum()
    union = w_0 * union_0 + w_1 * union_1

    dice = (2 * intersection + smoothing) / (union + smoothing)
    # fix nans
    dice[dice != dice] = dice.new_tensor([1.0])
    return 1 - dice.mean()

def bratsWeightedDiceLoss(outputs, labels):
    # bring outputs into correct shape
    wt, tc, et = outputs.chunk(3, dim=1)
    s = wt.shape
    wt = wt.view(s[0], s[2], s[3], s[4])
    tc = tc.view(s[0], s[2], s[3], s[4])
    et = et.view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask, tcMask, etMask = labels.chunk(3, dim=1)
    s = wtMask.shape
    wtMask = wtMask.view(s[0], s[2], s[3], s[4])
    tcMask = tcMask.view(s[0], s[2], s[3], s[4])
    etMask = etMask.view(s[0], s[2], s[3], s[4])

    # calculate losses
    wtWeightLoss = weightedDiceLoss(wt, wtMask)
    tcWeightLoss = weightedDiceLoss(tc, tcMask)
    etWeightLoss = weightedDiceLoss(et, etMask)

    return (wtWeightLoss + tcWeightLoss + etWeightLoss) / 5

def focalLoss(outputs, labels):

    alpha = 0.25
    gamma = 2.0
    pt_1 = torch.where(torch.eq(labels, 1), outputs, torch.ones_like(outputs))
    pt_0 = torch.where(torch.eq(labels, 0), outputs, torch.zeros_like(outputs))
    pt_1 = torch.clamp(pt_1, 1e-3, .999)
    pt_0 = torch.clamp(pt_0, 1e-3, .999)

    # return -torch.sum(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1))\
    #        -torch.sum((1 - alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))

    return -torch.sum(torch.log(pt_1)) - alpha * torch.sum(torch.pow(pt_0, gamma) * torch.log(1. - pt_0))

def bratsFocalLoss(outputs, labels):
    wt, tc, et = outputs.chunk(3, dim=1)
    s = wt.shape
    wt = wt.view(s[0], s[2], s[3], s[4])
    tc = tc.view(s[0], s[2], s[3], s[4])
    et = et.view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask, tcMask, etMask = labels.chunk(3, dim=1)
    s = wtMask.shape
    wtMask = wtMask.view(s[0], s[2], s[3], s[4])
    tcMask = tcMask.view(s[0], s[2], s[3], s[4])
    etMask = etMask.view(s[0], s[2], s[3], s[4])

    wtloss = focalLoss(wt, wtMask)
    tcloss = focalLoss(tc, tcMask)
    etloss = focalLoss(et, etMask)

    return (wtloss + tcloss + etloss) / 10

def bratsMixedLoss(outputs, labels, alpha=0.00001):
    return alpha * bratsFocalLoss(outputs, labels) + bratsDiceLoss(outputs, labels)

def bratsDiceLossOriginal5(outputs, labels, nonSquared=False):
    outputList = list(outputs.chunk(5, dim=1))
    labelsList = list(labels.chunk(5, dim=1))
    totalLoss = 0
    for pred, target in zip(outputList, labelsList):
        totalLoss = totalLoss + diceLoss(pred, target, nonSquared=nonSquared)
    return totalLoss

def sensitivity(pred, target):
    predBin = (pred > 0.5).float()
    intersection = (predBin * target).sum()
    allPositive = target.sum()

    # special case for zero positives
    if allPositive == 0:
        return 1.0
    return (intersection / allPositive).item()

def specificity(pred, target):
    predBinInv = (pred <= 0.5).float()
    targetInv = (target == 0).float()
    intersection = (predBinInv * targetInv).sum()
    allNegative = targetInv.sum()
    return (intersection / allNegative).item()

def getHd95(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target):
        surDist1 = medpyMetrics.__surface_distances(pred, target)
        surDist2 = medpyMetrics.__surface_distances(target, pred)
        hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)
        return hd95
    else:
        # Edge cases that medpy cannot handle
        return -1

def getWTMask(labels):
    return (labels != 0).float()

def getTCMask(labels):
    return ((labels != 0) * (labels != 2)).float() # We use multiplication as AND

def getETMask(labels):
    return (labels == 4).float()

def GeneralizedDiceLoss(output, target, eps=1e-5, weight_type='square'): # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """
    # target = target.float()

    if target.dim() == 4:
        target[target == 4] = 3 # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1])  # [N,H,W,D] -> [N,4,H,W,D]

    output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:,...] # [class, N*H*W*D]

    target_sum = target.sum(-1) # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :',weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)
    # logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))

    return 1 - 2. * intersect_sum / denominator_sum

def expand_target(x, n_class, mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:,1,:,:,:] = (x == 1)
        xx[:,2,:,:,:] = (x == 2)
        xx[:,3,:,:,:] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:,0,:,:,:] = (x == 1)
        xx[:,1,:,:,:] = (x == 2)
        xx[:,2,:,:,:] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)

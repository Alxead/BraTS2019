from comet_ml import Experiment, ExistingExperiment
import sys
sys.path.append("..")
import torch
import torch.optim as optim
import torch.nn as nn
import bratsUtils
import torch.nn.functional as F
import random

from torch.nn import init

id = random.getrandbits(64)

#restore experiment
#VALIDATE_ALL = False
PREDICT = True
####
RESTORE_ID = 10667378141858235118
RESTORE_EPOCH = 46
#LOG_COMETML_EXISTING_EXPERIMENT = ""

#general settings
SAVE_CHECKPOINTS = True #set to true to create a checkpoint at every epoch
EXPERIMENT_TAGS = ["bugfreeFinalDrop"]
EXPERIMENT_NAME = "subtract"
EPOCHS = 1000
BATCH_SIZE = 1
VIRTUAL_BATCHSIZE = 1
VALIDATE_EVERY_K_EPOCHS = 1
INPLACE = True

#hyperparameters
CHANNELS = 30
INITIAL_LR = 1e-4
L2_REGULARIZER = 1e-5
HYPERCOLUMN = False

#logging settings
LOG_EVERY_K_ITERATIONS = 30 #0 to disable logging
LOG_MEMORY_EVERY_K_ITERATIONS = False
LOG_MEMORY_EVERY_EPOCH = True
LOG_EPOCH_TIME = True
LOG_VALIDATION_TIME = True
LOG_HAUSDORFF_EVERY_K_EPOCHS = 0 #must be a multiple of VALIDATE_EVERY_K_EPOCHS
LOG_COMETML = False
LOG_PARAMCOUNT = True
LOG_LR_EVERY_EPOCH = True

#data and augmentation
TRAIN_ORIGINAL_CLASSES = False #train on original 5 classes
DATASET_WORKERS = 1
SOFT_AUGMENTATION = False #Soft augmetation directly works on the 3 classes. Hard augmentation augments on the 5 orignal labels, then takes the argmax
NN_AUGMENTATION = True #Has priority over soft/hard augmentation. Uses nearest-neighbor interpolation
DO_ROTATE = True
DO_SCALE = True
DO_FLIP = True
DO_ELASTIC_AUG = True
DO_INTENSITY_SHIFT = True
RANDOM_CROP = [128, 128, 128]

# TRAIN_ORIGINAL_CLASSES = False #train on original 5 classes
# DATASET_WORKERS = 1
# SOFT_AUGMENTATION = False #Soft augmetation directly works on the 3 classes. Hard augmentation augments on the 5 orignal labels, then takes the argmax
# NN_AUGMENTATION = True #Has priority over soft/hard augmentation. Uses nearest-neighbor interpolation
# DO_ROTATE = False
# DO_SCALE = False
# DO_FLIP = False
# DO_ELASTIC_AUG = False
# DO_INTENSITY_SHIFT = False
# RANDOM_CROP = [128, 128, 128]

ROT_DEGREES = 20
SCALE_FACTOR = 1.1
SIGMA = 10
MAX_INTENSITY_SHIFT = 0.1

#ideas
DO_MIXUP = False
AVERAGE_DATA = True

if LOG_COMETML:
    if not "LOG_COMETML_EXISTING_EXPERIMENT" in locals():
        experiment = Experiment(api_key="", project_name="", workspace="")
    else:
        experiment = ExistingExperiment(api_key="", previous_experiment=LOG_COMETML_EXISTING_EXPERIMENT, project_name="", workspace="")
else:
    experiment = None

#network funcitons
if TRAIN_ORIGINAL_CLASSES:
    loss = bratsUtils.bratsDiceLossOriginal5
else:
    #loss = bratsUtils.bratsDiceLoss
    def loss(outputs, labels):
        # return bratsUtils.bratsDiceLoss(outputs, labels, nonSquared=True)
        # return bratsUtils.bratsFocalLoss(outputs, labels)
        # return bratsUtils.bratsMixedLoss(outputs, labels)
        return bratsUtils.bratsWeightedDiceLoss(outputs, labels)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class EncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, maxpool=False, secondConv=True, hasDropout=False):
        super(EncoderModule, self).__init__()
        groups = min(outChannels, CHANNELS)
        self.maxpool = maxpool
        self.secondConv = secondConv
        self.hasDropout = hasDropout
        self.conv1 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, outChannels)
        if secondConv:
            self.conv2 = nn.Conv3d(outChannels, outChannels, 3, padding=1, bias=False)
            self.gn2 = nn.GroupNorm(groups, outChannels)
        if hasDropout:
            self.dropout = nn.Dropout3d(0.2, True)

    def forward(self, x):
        if self.maxpool:
            x = F.max_pool3d(x, 2)
        doInplace = INPLACE and not self.hasDropout
        x = F.leaky_relu(self.gn1(self.conv1(x)), inplace=doInplace)
        if self.hasDropout:
            x = self.dropout(x)
        if self.secondConv:
            x = F.leaky_relu(self.gn2(self.conv2(x)), inplace=INPLACE)
        return x

class DecoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, upsample=False, firstConv=True):
        super(DecoderModule, self).__init__()
        groups = min(outChannels, CHANNELS)
        self.upsample = upsample
        self.firstConv = firstConv
        if firstConv:
            self.conv1 = nn.Conv3d(inChannels, inChannels, 3, padding=1, bias=False)
            self.gn1 = nn.GroupNorm(groups, inChannels)
        self.conv2 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, outChannels)

    def forward(self, x):
        if self.firstConv:
            x = F.leaky_relu(self.gn1(self.conv1(x)), inplace=INPLACE)
        x = F.leaky_relu(self.gn2(self.conv2(x)), inplace=INPLACE)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return x


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(ch_out)
        self.conv2 = nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(ch_out)
            )
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.leaky_relu(out)
        return out

# Hypercolumns
class HyperColumn(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=1):
        super(HyperColumn, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
             for in_ch, out_ch in zip(input_channels, output_channels)])
        # self.up = nn.Upsample(tuple(im_size), mode='trilinear', align_corners=False)

        self.up0 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=False)
        self.up1 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=1, mode='trilinear', align_corners=False)

    def forward(self, xs, last_layer=None):

        # hcs = [self.up(c(x)) for c, x in zip(self.convs, xs)]
        cs = [c(x) for c, x in zip(self.convs, xs)]
        hcs = []
        hcs.append(self.up0(cs[0]))
        hcs.append(self.up1(cs[1]))
        hcs.append(self.up2(cs[2]))
        hcs.append(self.up3(cs[3]))
        hcs.append(self.up4(cs[4]))

        if last_layer is not None:
            hcs.append(last_layer)
        return torch.cat(hcs, dim=1)

class NoNewNet(nn.Module):
    def __init__(self):
        super(NoNewNet, self).__init__()
        channels = CHANNELS
        self.levels = 5
        self.useHyperColumn = HYPERCOLUMN

        #create encoder levels
        encoderModules = []
        encoderModules.append(EncoderModule(4, channels, False, True))
        for i in range(self.levels - 2):
            encoderModules.append(EncoderModule(channels * pow(2, i), channels * pow(2, i+1), True, True))
        encoderModules.append(EncoderModule(channels * pow(2, self.levels - 2), channels * pow(2, self.levels - 1), True, False))
        self.encoders = nn.ModuleList(encoderModules)

        # freeze the encoder
        # for p in self.parameters():
        #     p.requires_grad = False

        #create decoder levels
        decoderModules = []
        decoderModules.append(DecoderModule(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 2), True, False))
        for i in range(self.levels - 2):
            decoderModules.append(DecoderModule(channels * pow(2, self.levels - i - 2), channels * pow(2, self.levels - i - 3), True, True))
        decoderModules.append(DecoderModule(channels, channels, False, True))
        self.decoders = nn.ModuleList(decoderModules)

        if self.useHyperColumn:
            hc_ch_in, hc_ch_out = [480, 240, 120, 60, 30], [16] * 5
            self.hyperColumn = HyperColumn(hc_ch_in, hc_ch_out)
            self.drop = nn.Dropout3d(p=0.5, inplace=True)
            self.lastConv = nn.Conv3d(channels+sum(hc_ch_out)+4, 3, 1, bias=True)
            self.resBlock = ResBlk(CHANNELS+4, CHANNELS+4)
        else:
            self.lastConv = nn.Conv3d(channels, 3, 1, bias=True)

    def forward(self, x):
        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)

        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)
        x = torch.sigmoid(x)
        return x

    # def forward(self, input_image):
    #
    #     inputStack = []
    #     x = self.encoders[0](input_image)
    #     inputStack.append(x)
    #     x = self.encoders[1](x)
    #     inputStack.append(x)
    #     x = self.encoders[2](x)
    #     inputStack.append(x)
    #     x = self.encoders[3](x)
    #     inputStack.append(x)
    #     x = self.encoders[4](x)
    #     z0 = x                                      # z0 [480,8,8,8]
    #
    #     x = self.decoders[0](x)
    #     z1 = x                                      # z1 [240,16,16,16]
    #
    #     x = x + inputStack.pop()
    #     x = self.decoders[1](x)
    #     z2 = x                                      # z2 [120,32,32,32]
    #
    #     x = x + inputStack.pop()
    #     x = self.decoders[2](x)
    #     z3 = x                                      # z3 [60,64,64,64]
    #
    #     x = x + inputStack.pop()
    #     x = self.decoders[3](x)
    #     z4 = x                                      # z4 [30,128,128,128]
    #
    #     x = x + inputStack.pop()
    #     x = self.decoders[4](x)
    #
    #     if self.useHyperColumn:
    #         z5 = x  # z5 [30,128,128,128]
    #         z5 = torch.cat((z5, input_image), dim=1)
    #         z5 = self.resBlock(z5)
    #         x = self.hyperColumn([z0, z1, z2, z3, z4], z5)
    #         x = self.drop(x)
    #
    #     x = self.lastConv(x)
    #     x = torch.sigmoid(x)
    #     return x

net = NoNewNet()

def trainableParam(named_parameters):

    parameters=[]
    for name, param in named_parameters:
        if 'decoders' in name or 'lastConv' in name:
            parameters.append(param)
    return parameters

# parameters = trainableParam(net.named_parameters())
# optimizer = optim.Adam(net.parameters(), lr=INITIAL_LR, weight_decay=L2_REGULARIZER)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=INITIAL_LR, weight_decay=L2_REGULARIZER)
lr_sheudler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 400, 550], 0.2)
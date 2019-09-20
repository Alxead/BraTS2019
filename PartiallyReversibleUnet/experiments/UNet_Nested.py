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
PREDICT = False
#RESTORE_ID = 395
#RESTORE_EPOCH = 350
#LOG_COMETML_EXISTING_EXPERIMENT = ""

#general settings
SAVE_CHECKPOINTS = True #set to true to create a checkpoint at every epoch
EXPERIMENT_TAGS = ["bugfreeFinalDrop"]
EXPERIMENT_NAME = "UNet++"
EPOCHS = 1000
BATCH_SIZE = 1
VIRTUAL_BATCHSIZE = 1
VALIDATE_EVERY_K_EPOCHS = 1
INPLACE = True

#hyperparameters
CHANNELS = 30
INITIAL_LR = 1e-4
L2_REGULARIZER = 1e-5

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
DO_MIXUP = True

ROT_DEGREES = 20
SCALE_FACTOR = 1.1
SIGMA = 10
MAX_INTENSITY_SHIFT = 0.1

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
        return bratsUtils.bratsDiceLoss(outputs, labels, nonSquared=True)

# initalize the module
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

class unetConv3(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv3, self).__init__()
        self.n = n

        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, stride, padding),
                                     nn.BatchNorm3d(out_size),
                                     nn.ReLU(inplace=True))
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size
        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, stride, padding),
                                     nn.ReLU(inplace=True))
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        for m in self.children():
            init_weights(m, 'kaiming')

    def forward(self, input):
        x = input
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

class unetUp(nn.Module):

    def __init__(self, in_size, out_size, is_deconv=False, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv3(in_size+(n_concat-2)*out_size, out_size, is_batchnorm=False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(in_size, out_size, kernel_size=1)
            )

        for m in self.children():
            if m.__class__.__name__.find('unetConv3') != -1:continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):

        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)

class UNet_Nested(nn.Module):

    def __init__(self, in_channels=4, n_classes=3, is_deconv=False, is_batchnorm=True, is_ds=False):

        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds                     # deep supervision

        filters = [20, 40, 80, 160, 320]

        # downsampling
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.conv00 = unetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv3(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv3(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv3(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv3d(filters[0], n_classes, 1)
        self.sigmoid = nn.Sigmoid()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        x00 = self.conv00(inputs)
        maxpool0 = self.maxpool(x00)
        x10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool(x10)
        x20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool(x20)
        x30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool(x30)
        x40 = self.conv40(maxpool3)
        # column : 1
        x01 = self.up_concat01(x10, x00)
        x11 = self.up_concat11(x20, x10)
        x21 = self.up_concat21(x30, x20)
        x31 = self.up_concat31(x40, x30)
        # column : 2
        x02 = self.up_concat02(x11, x00, x01)
        x12 = self.up_concat12(x21, x10, x11)
        x22 = self.up_concat22(x31, x20, x21)
        # column : 3
        x03 = self.up_concat03(x12, x00, x01, x02)
        x13 = self.up_concat13(x22, x10, x11, x12)
        # column : 4
        x04 = self.up_concat04(x13, x00, x01, x02, x03)

        # final layer
        final_1 = self.final_1(x01)
        final_2 = self.final_2(x02)
        final_3 = self.final_3(x03)
        final_4 = self.final_4(x04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return self.sigmoid(final)
        else:
            return self.sigmoid(final_4)

net = UNet_Nested()

optimizer = optim.Adam(net.parameters(), lr=INITIAL_LR, weight_decay=L2_REGULARIZER)
lr_sheudler = optim.lr_scheduler.MultiStepLR(optimizer, [250, 400, 550], 0.2)

optimizer = optim.Adam(filter())
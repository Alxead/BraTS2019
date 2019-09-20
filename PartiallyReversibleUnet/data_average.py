import os
import torch
import bratsDataset
import segmenter
import systemsetup
import matplotlib.pyplot as plt
from dataProcessing import utils
import numpy as np

import experiments.noNewNet as expConfig

trainset = bratsDataset.BratsDataset(systemsetup.BRATS_PATH, expConfig, mode="train")
valset = bratsDataset.BratsDataset(systemsetup.BRATS_PATH, expConfig, mode="validation")

# sum_inputs = 0
# count = 0
# for i in range(len(trainset)):
#     inputs, pid, labels = trainset[i]
#     print("processing no.{} {}".format(i, pid))
#     inputs = inputs.numpy()
#     sum_inputs += inputs
#     count += 1
# print(count)
#
# for i in range(len(valset)):
#     inputs, pid, labels = valset[i]
#     print("processing no.{} {}".format(i, pid))
#     inputs = inputs.numpy()
#     sum_inputs += inputs
#     count += 1
# print(count)
#
# average_inputs = sum_inputs / count
#
# basePath = '/home/liujing/data/MICCAI_BraTS/2019/training'
# path = os.path.join(basePath, "MixupData.nii.gz")
# utils.save_nii(path, average_inputs, None, None)

basePath = '/home/liujing/data/MICCAI_BraTS/2019/training/MixupData.nii.gz'
img, _, _ = utils.load_nii(basePath)
img = np.array(img)

for i in range(4):
    average_image = img[i, :, :, 80]
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    plt.imshow(average_image, cmap='gray')
    basePath = '/home/liujing/data/MICCAI_BraTS/2019/training'
    plt.savefig(f'{basePath}/mixup' + str(i) + '.png')
    plt.show()




import os
import torch
import numpy as np
from dataProcessing import utils

basePath = '/home/liujing/hyd/PartiallyReversibleUnet/predict'
ensembledPath = '/home/liujing/hyd/PartiallyReversibleUnet/ensembled_predict'

pathlist = []
for cur_dir, dirs, files in os.walk(basePath):
    for dir in dirs:
        if dir.endswith('fullsize'):
            pathlist.append(os.path.join(basePath, dir))

print(pathlist)
print(len(pathlist))

filelist = []
for file in os.listdir(pathlist[0]):
    filelist.append(file)

print(len(filelist))
filelist.remove('BraTS19_TCIA03_318_1.nii.gz')  #BraTS19_TCIA03_318_1
print(len(filelist))
print(filelist)

count = 0
for file in filelist:
    print("No.{} ensemble processing {}".format(count, file))
    count += 1
    filePathList = []
    for path in pathlist:
        filePath = os.path.join(path, file)
        filePathList.append(filePath)

    wt_total = torch.zeros(1, 1, 240, 240, 155)
    tc_total = torch.zeros(1, 1, 240, 240, 155)
    et_total = torch.zeros(1, 1, 240, 240, 155)
    for filePath in filePathList:
        fullsize, _, _ = utils.load_nii(filePath)
        fullsize = torch.from_numpy(fullsize)
        wt, tc, et = fullsize.chunk(3, dim=1)
        wt_total += wt
        tc_total += tc
        et_total += et

    file_number = len(filePathList)
    wt = wt_total / file_number
    tc = tc_total / file_number
    et = et_total / file_number
    wt = (wt > 0.5).view(240, 240, 155)
    tc = (tc > 0.5).view(240, 240, 155)
    et = (et > 0.5).view(240, 240, 155)

    result = torch.zeros_like(wt, dtype=torch.uint8)
    result[wt] = 2
    result[tc] = 1
    result[et] = 4

    npResult = result.cpu().numpy()
    ET_voxels = (npResult == 4).sum()
    if ET_voxels < 500:
        npResult[np.where(npResult == 4)] = 1

    path = os.path.join(ensembledPath, file)
    utils.save_nii(path, npResult, None, None)
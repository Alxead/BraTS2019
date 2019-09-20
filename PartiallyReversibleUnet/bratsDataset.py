import torch
import torch.utils.data
import h5py
import numpy as np
import time
import random
import dataProcessing.augmentation as aug

class BratsDataset(torch.utils.data.Dataset):
    # mode must be trian, test or val
    def __init__(self, filePath, expConfig, mode="train", randomCrop=None, hasMasks=True, returnOffsets=False, average_data=None):
        super(BratsDataset, self).__init__()
        self.filePath = filePath
        self.mode = mode
        self.file = None
        self.trainOriginalClasses = expConfig.TRAIN_ORIGINAL_CLASSES
        self.randomCrop = randomCrop
        self.hasMasks = hasMasks
        self.returnOffsets = returnOffsets
        self.average_data = average_data

        # augmentation settings
        self.nnAugmentation = expConfig.NN_AUGMENTATION
        self.softAugmentation = expConfig.SOFT_AUGMENTATION
        self.doRotate = expConfig.DO_ROTATE
        self.rotDegrees = expConfig.ROT_DEGREES
        self.doScale = expConfig.DO_SCALE
        self.scaleFactor = expConfig.SCALE_FACTOR
        self.doFlip = expConfig.DO_FLIP
        self.doElasticAug = expConfig.DO_ELASTIC_AUG
        self.sigma = expConfig.SIGMA
        self.doIntensityShift = expConfig.DO_INTENSITY_SHIFT
        self.maxIntensityShift = expConfig.MAX_INTENSITY_SHIFT
        self.doMixUp = expConfig.DO_MIXUP


    def __getitem__(self, index):

        # lazily open file
        self.openFileIfNotOpen()

        # deal with average_data
        if not self.average_data is None:
            average_data_return = np.transpose(self.average_data, (1, 2, 3, 0))

        # load from hdf5 file
        image = self.file["images_" + self.mode][index, ...]
        if self.hasMasks: labels = self.file["masks_" + self.mode][index, ...]

        # Prepare data depeinding on soft/hard augmentation scheme
        if not self.nnAugmentation:
            if not self.trainOriginalClasses and (self.mode != "train" or self.softAugmentation):
                if self.hasMasks: labels = self._toEvaluationOneHot(labels)
                defaultLabelValues = np.zeros(3, dtype=np.float32)
            else:
                if self.hasMasks: labels = self._toOrignalCategoryOneHot(labels)
                defaultLabelValues = np.asarray([1, 0, 0, 0, 0], dtype=np.float32)
        elif self.hasMasks:
            if labels.ndim < 4:
                labels = np.expand_dims(labels, 3)
            defaultLabelValues = np.asarray([0], dtype=np.float32)

        if self.nnAugmentation:
            if self.hasMasks: labels = self._toEvaluationOneHot(np.squeeze(labels, 3))
        else:
            if self.mode == "train" and not self.softAugmentation and not self.trainOriginalClasses and self.hasMasks:
                labels = self._toOrdinal(labels)
                labels = self._toEvaluationOneHot(labels)

        if self.mode == 'train' and self.doMixUp:
            datasize = self.file["images_" + self.mode].shape[0]
            idx = np.random.randint(0, datasize)
            image2 = self.file["images_" + self.mode][idx, ...]
            labels2 = self.file["masks_" + self.mode][idx, ...]

            # Prepare data depeinding on soft/hard augmentation scheme
            if not self.nnAugmentation:
                if not self.trainOriginalClasses and (self.mode != "train" or self.softAugmentation):
                    if self.hasMasks: labels2 = self._toEvaluationOneHot(labels2)
                    defaultLabelValues = np.zeros(3, dtype=np.float32)
                else:
                    if self.hasMasks: labels2 = self._toOrignalCategoryOneHot(labels2)
                    defaultLabelValues = np.asarray([1, 0, 0, 0, 0], dtype=np.float32)
            elif self.hasMasks:
                if labels2.ndim < 4:
                    labels2 = np.expand_dims(labels2, 3)
                defaultLabelValues = np.asarray([0], dtype=np.float32)

            if self.nnAugmentation:
                if self.hasMasks: labels2 = self._toEvaluationOneHot(np.squeeze(labels2, 3))
            else:
                if self.mode == "train" and not self.softAugmentation and not self.trainOriginalClasses and self.hasMasks:
                    labels2 = self._toOrdinal(labels2)
                    labels2 = self._toEvaluationOneHot(labels2)

            # alpha = 1.0  # Hyperparameter
            # m = 0.3      # Hyperparameter
            # lam = np.random.beta(alpha, alpha)
            # image = lam * image + (1 - lam) * image2
            #
            # target = np.zeros_like(labels)
            # if lam > m:
            #     target = target + labels
            # if (1 - lam) > m:
            #     target = target + labels2
            # target[target > 1] = 1
            # labels = target

            m1 = 0.4
            m2 = 0.2
            m3 = 0.1
            alpha = 1.0
            lam = np.random.beta(alpha, alpha)
            image = lam * image + (1 - lam) * image2

            target = np.zeros_like(labels)
            # if lam > m1:
            #     target[..., 0] = target[..., 0] + labels[..., 0]
            # if (1 - lam) > m1:
            #     target[..., 0] = target[..., 0] + labels2[..., 0]

            target[..., 0] = lam * target[..., 0] + (1-lam) * labels2[..., 0]

            if lam > m2:
                target[..., 1] = target[..., 1] + labels[..., 1]
            if (1 - lam) > m2:
                target[..., 1] = target[..., 1] + labels2[..., 1]

            if lam > m3:
                target[..., 2] = target[..., 2] + labels[..., 2]
            if (1 - lam) > m3:
                target[..., 2] = target[..., 2] + labels2[..., 2]

            target[target == 2] = 1
            labels = target

        # augment data
        if self.mode == "train":
            if self.average_data is None:
                image, labels = aug.augment3DImage(image,
                                                   labels,
                                                   defaultLabelValues,
                                                   self.nnAugmentation,
                                                   self.doRotate,
                                                   self.rotDegrees,
                                                   self.doScale,
                                                   self.scaleFactor,
                                                   self.doFlip,
                                                   self.doElasticAug,
                                                   self.sigma,
                                                   self.doIntensityShift,
                                                   self.maxIntensityShift)
            else:
                image, average_data_return, labels = aug.augment_two3DImage(image,
                                                                            average_data_return,
                                                                            labels,
                                                                            defaultLabelValues,
                                                                            self.nnAugmentation,
                                                                            self.doRotate,
                                                                            self.rotDegrees,
                                                                            self.doScale,
                                                                            self.scaleFactor,
                                                                            self.doFlip,
                                                                            self.doElasticAug,
                                                                            self.sigma,
                                                                            self.doIntensityShift,
                                                                            self.maxIntensityShift)
        # random crop
        if not self.randomCrop is None:
            shape = image.shape
            x = random.randint(0, shape[0] - self.randomCrop[0])
            y = random.randint(0, shape[1] - self.randomCrop[1])
            z = random.randint(0, shape[2] - self.randomCrop[2])
            image = image[x:x+self.randomCrop[0], y:y+self.randomCrop[1], z:z+self.randomCrop[2], :]
            if self.hasMasks: labels = labels[x:x + self.randomCrop[0], y:y + self.randomCrop[1], z:z + self.randomCrop[2], :]
            if not self.average_data is None:
                average_data_return = average_data_return[x:x + self.randomCrop[0], y:y + self.randomCrop[1], z:z + self.randomCrop[2], :]

        image = np.transpose(image, (3, 0, 1, 2))  # bring into NCWH format
        if not self.average_data is None:
            average_data_return = np.transpose(average_data_return, (3, 0, 1, 2))
        if self.hasMasks: labels = np.transpose(labels, (3, 0, 1, 2))  # bring into NCWH format

        # to tensor
        # image = image[:, 0:32, 0:32, 0:32]
        image = torch.from_numpy(image)
        if not self.average_data is None:
            average_data_return = torch.from_numpy(average_data_return)

        if self.hasMasks:
            # labels = labels[:, 0:32, 0:32, 0:32]
            labels = torch.from_numpy(labels) 

        # get pid
        pid = self.file["pids_" + self.mode][index]

        if self.returnOffsets:
            xOffset = self.file["xOffsets_" + self.mode][index]
            yOffset = self.file["yOffsets_" + self.mode][index]
            zOffset = self.file["zOffsets_" + self.mode][index]
            if self.hasMasks:
                if not self.average_data is None:
                    return image, str(pid), labels, average_data_return, xOffset, yOffset, zOffset
                return image, str(pid), labels, xOffset, yOffset, zOffset
            else:
                if not self.average_data is None:
                    return image, str(pid), average_data_return, xOffset, yOffset, zOffset
                return image, pid, xOffset, yOffset, zOffset
        else:
            if self.hasMasks:
                if not self.average_data is None:
                    return image, str(pid), labels, average_data_return
                return image, str(pid), labels
            else:
                return image, pid

    def __len__(self):
        # lazily open file
        self.openFileIfNotOpen()

        return self.file["images_" + self.mode].shape[0]

    def openFileIfNotOpen(self):
        if self.file == None:
            self.file = h5py.File(self.filePath, "r")

    def _toEvaluationOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], 3], dtype=np.float32)
        out[:, :, :, 0] = (labels != 0)                          # WT
        out[:, :, :, 1] = (labels != 0) * (labels != 2)          # TC
        out[:, :, :, 2] = (labels == 4)                          # ET
        return out

    def _toOrignalCategoryOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], 5], dtype=np.float32)
        for i in range(5):
            out[:, :, :, i] = (labels == i)
        return out

    def _toOrdinal(self, labels):
        return np.argmax(labels, axis=3)

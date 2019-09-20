# import comet_ml
import torch
import bratsDataset
import segmenter
import systemsetup
from dataProcessing import utils

# import experiments.noNewReversible as expConfig
# import experiments.noNewReversibleFat as expConfig
# import experiments.noNewNet as expConfig
# import experiments.noNewNet5 as expConfig
# import experiments.UNet_Nested as expConfig
# import experiments.DMFNet as expConfig
# import experiments.atlas_noNewNet2 as expConfig
# import experiments.atlas_noNewNet_cat as expConfig
import experiments.atlas_noNewNet_subtract as expConfig
# import experiments.atlas_noNewNet as expConfig

class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

def main():

    #  setup experiment logging to comet.ml
    if expConfig.LOG_COMETML:
        hyper_params = {"experimentName": expConfig.EXPERIMENT_NAME,
                        "epochs": expConfig.EPOCHS,
                        "batchSize": expConfig.BATCH_SIZE,
                        "channels": expConfig.CHANNELS,
                        "virualBatchsize": expConfig.VIRTUAL_BATCHSIZE}
        expConfig.experiment.log_parameters(hyper_params)
        expConfig.experiment.add_tags([expConfig.EXPERIMENT_NAME, "ID{}".format(expConfig.id)])
        if hasattr(expConfig, "EXPERIMENT_TAGS"): expConfig.experiment.add_tags(expConfig.EXPERIMENT_TAGS)
        print(bcolors.OKGREEN + "Logging to comet.ml" + bcolors.ENDC)
    else:
        print(bcolors.WARNING + "Not logging to comet.ml" + bcolors.ENDC)

    # log parameter count
    if expConfig.LOG_PARAMCOUNT:
        paramCount = sum(p.numel() for p in expConfig.net.parameters() if p.requires_grad)
        print("Parameters: {:,}".format(paramCount).replace(",", "'"))

    # load data
    if expConfig.AVERAGE_DATA:
        average_data, _, _ = utils.load_nii("/home/liujing/data/MICCAI_BraTS/2019/training/MixupData.nii.gz")
    else:
        average_data = None
    randomCrop = expConfig.RANDOM_CROP if hasattr(expConfig, "RANDOM_CROP") else None
    trainset = bratsDataset.BratsDataset(systemsetup.BRATS_PATH, expConfig, mode="train", randomCrop=randomCrop, average_data=average_data)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=expConfig.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=expConfig.DATASET_WORKERS)

    valset = bratsDataset.BratsDataset(systemsetup.BRATS_PATH, expConfig, mode="validation", average_data=average_data)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, pin_memory=True, num_workers=expConfig.DATASET_WORKERS)

    challengeValset = bratsDataset.BratsDataset(systemsetup.BRATS_VAL_PATH, expConfig, mode="validation", hasMasks=False, returnOffsets=True, average_data=average_data)
    challengeValloader = torch.utils.data.DataLoader(challengeValset, batch_size=1, shuffle=False, pin_memory=True, num_workers=expConfig.DATASET_WORKERS)

    seg = segmenter.Segmenter(expConfig, trainloader, valloader, challengeValloader, trainset)
    if hasattr(expConfig, "VALIDATE_ALL") and expConfig.VALIDATE_ALL:
        seg.validateAllCheckpoints()
    elif hasattr(expConfig, "PREDICT") and expConfig.PREDICT:
        seg.makePredictions()
    else:
        seg.train(is_mixup=False)

if __name__ == "__main__":
    main()
# BraTS2019 hyd‘s most expriments
The main program's framework and code refers to this repository [PartiallyReversibleUnet](<https://github.com/RobinBruegger/PartiallyReversibleUnet>)

the idea I've tried:

- UNet++ : (UNet_Nested.py)

- mixup(whole picture)

- tumor-mixup

- hardlabel-mixup (Asymmetric mixup)

- hypercolumn

- focalloss

- general dice loss

- downsample from 4 times to 5 times

- [DMFNet](<https://github.com/China-LiuXiaopeng/BraTS-DMFNet>) (use his pre-trained model, but it did not work well) 

- some expriments using the brain map(the average of all MRI data)

  - add a branch(atlas_noNewNet.py)
  - concat input and map
  - input subtract map

- ensemble(ensemble_prediction.py)（the average of the predictions, works well!)

- visualization.py(show the image on tensorboard during traing)

  

## Virtual Environment Setup
The code is implemented in Python 3.6 using PyTorch 1.1.0. Follow the steps below to install all dependencies:
* Set up a virtual environment (e.g. conda or virtualenv) with Python 3.6
* Install all non-PyTorch requirements using: `pip install -r dataProcessing/brats18_data_loader.py`
* Install PyTorch following the instructions on ther [website](https://pytorch.org/).

## Data
We trained with the [BraTS 2018 dataset](https://www.med.upenn.edu/sbia/brats2018/data.html), which is available from the organizers of the [BraTS challenge](https://www.med.upenn.edu/sbia/brats2018.html).

To prepare the data, adjust the paths at the end of `dataProcessing/brats18_data_loader.py`. Then run this script. Do the same for `dataProcessing/brats18_validation_data_loader.py`, which prepares the validation data.
	
## Running the code
* Adjust the path in `systemsetup.py` to match your system.
* Run the `train.py` script

The settings for the experiments are each in an individual file located in the `experiments/` folder.
You can change the experiment by importing a different experiment in the file `segmenter.py`.

## Creating checkpoits and prediction
* To create a checkpoint after every epoch, set `SAVE_CHECKPOINTS = True` in the corresponding experiment file
* For inference, you need to load a checkpoint form a previously trained model. To achieve this, set the following three fields in the corresponding experiment file:
```
PREDICT = True
RESTORE_ID = <Id to load>
RESTORE_EPOCH = <Epoch to load>
```
Running `train.py` now will run inference for all images in the validation dataset.
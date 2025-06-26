# ABFL

for segmentation tasks

```txt
 ____________
|            |
|  Model 1   | __ mask __
|____________|             \
 ____________              ______________
|            |            |              |
|  Model 2   |----mask----|  Attention   | -------> improved segmentation mask 
|____________|            |______________|
 ____________              /
|            | __ mask __ /
|  Model 3   |
|____________|

```

Base Models had to be trained exactly the same way, with the same dataloaders, same transforms, same hyperparameters.

For hps:

```txt
BATCHSIZE = 1
MAXEPOCHS = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.99
```

Should incorporate an LR scheduler. Hyperparameters based on Unet paper [https://arxiv.org/abs/1505.04597].

For transforms look at ```data\aug.py```. Geometric and photometric augmentations separated for faster data processing.

For dataloaders look at ```data\dataloaders.py```.

For base model implementations look at ```models\*```

For controlled variables and repo configuration look at ```config.py```

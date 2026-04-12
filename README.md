# ABFL

Code pertaining to Masked Attention for cross model fusion. 

Quick little schema of things:

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

Will make sure to attach a more detailed diagram soon.

Model architectures benchmarked under similar hyperparameters and comparable network depth and size.

Fusion methods benchmarked: 
1. Cross model pixel-wise mean (Arithmetic, Geometric, ...)
2. Weighted cross Model fusion
3. Convolutional fusion (akin to boosting, slapping a CNN on output heads of previous segmentations)
4. 

Should incorporate an LR scheduler. Hyperparameters based on Unet paper [https://arxiv.org/abs/1505.04597].

For transforms look at ```data\aug.py```. Geometric and photometric augmentations separated for faster data processing.

For dataloaders look at ```data\dataloaders.py```.

For base model implementations look at ```models\*```. Most models come from [SMP](https://smp.readthedocs.io/en/latest/models.html#unetplusplus). Some come from Hugging Face.

For specific net configurations, check out ```models/config.py```.

> Some caveats that are annoying, but will take too long to refactor: 

CUB has a lot of custom code for loading and transforms. Pascal VOC, ADE20K, Cityscapes dsets don't.


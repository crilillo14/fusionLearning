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


Fusion methods benchmarked: 
1. Cross model pixel-wise mean (Arithmetic, Geometric, ...)
2. Weighted cross Model fusion
3. Convolutional fusion (akin to boosting, slapping a CNN on output heads of previous segmentations)
4. 

Should incorporate an LR scheduler. Hyperparameters based on Unet paper [https://arxiv.org/abs/1505.04597].

For transforms look at ```data\aug.py```. Geometric and photometric augmentations separated for faster data processing.

For dataloaders look at ```data\dataloaders.py```.

For base model implementations look at ```models\*```. Most models come from [SMP](https://smp.readthedocs.io/en/latest/models.html#unetplusplus)

For specific net configurations, check out ```models/config.py```.


Some caveats that are annoying, but will take too long to refactor: 

Everything is imported from the inner fusionLearning package, and I haven't structured paths and modules as actual importable packages. TODO.

CUB has a lot of custom code for loading and transforms. Other datasets don't. 


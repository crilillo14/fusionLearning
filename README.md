### ABFL

Where base models could be ViTs, Unets, etc.

```
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

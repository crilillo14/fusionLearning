
# Base Model Pipeline

## Initializing venv

Make sure to have the same venv for all base models [ Python 3.13.3 ]

> If working in a jupyter notebook, make sure to have the same kernel for all base models. Also change cwd to root.

## Declare macros

Outlined in CONSTVARS.txt

## Initialize dataloaders

initialize train, validation and test dataloaders with ```dataloaders.py```

## Initialize model

initialize model with smp, loss with torch, optimizer with torch, metrics with torchmetrics

```torch.compile()``` for improved kernel performance

## Training

add to it later...

## Testing

add to it later...


# design philosophy, stack, considerations.

Ok, so usually I'd go ahead and implement a model class that defines the layers, forward, and backward fns. 

But with [ smp ] (), dont need to. Check the docs and examples. Unet > simple.ipynb

__Vital libraries needed >__

## Preprocessing 

torchvision, PIL

## Training step 

smp, 
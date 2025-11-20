import torch
import time
import segmentation_models_pytorch as smp

weights = torch.load("weights/Unet_resnet34_10E/best_model.pth")  # your model (e.g. UNet)

model = smp.Unet(
    encoder_name="resnet34",  
    encoder_weights=None,  
    in_channels=3,  
    classes=2,
)

model.load_state_dict(weights)

model = model.to("cuda")
comp = torch.compile(model)

# dummy input
x = torch.randn(1, 3, 224, 224).cuda()

# warm up
for _ in range(10):
    _ = comp(x)

# time compiled
start = time.time()
for _ in range(50):
    _ = comp(x)
torch.cuda.synchronize()
print("Compiled model time:", time.time() - start)

# time eager
start = time.time()
for _ in range(50):
    _ = model(x)
torch.cuda.synchronize()
print("Eager model time:", time.time() - start)

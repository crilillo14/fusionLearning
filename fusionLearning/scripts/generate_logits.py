import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import Unet, UnetPlusPlus, Linknet, FPN # Assuming these are in models/unet.py
from data.dataloaders import CUBDataset # Assuming a dataloader

# --- Configuration ---
# This should be adapted to your project structure
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'CUB_200_2011')
BASE_MODELS_DIR = os.path.join(ROOT_DIR, 'baseModels')
LOGITS_DIR = os.path.join(ROOT_DIR, 'data', 'logits')
NUM_CLASSES = 2 # Background/Foreground

# Define the models to generate logits from
ARCH_ENCODER_PAIRINGS = {
    'Unet': ["resnet34", "resnet18"],
    'UnetPlusPlus': ["resnet34", "resnet18"],
    'Linknet': ["resnet18"],
    'FPN': ["resnet34"],
}

ARCH_MAP = {
    'Unet': Unet,
    'UnetPlusPlus': UnetPlusPlus,
    'Linknet': Linknet,
    'FPN': FPN
}

def generate_logits_for_model(model, model_name, dataloader, device):
    model.to(device)
    model.eval()

    output_dir = os.path.join(LOGITS_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating logits for {model_name}...")
    for i, (images, _) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)

        # Save each logit tensor in the batch
        for j in range(logits.shape[0]):
            image_idx = i * dataloader.batch_size + j
            # Use a unique identifier from the dataset if available, otherwise use index
            image_id = dataloader.dataset.image_paths[image_idx].split('/')[-1].replace('.jpg', '')
            logit_path = os.path.join(output_dir, f"{image_id}.pt")
            torch.save(logits[j].cpu(), logit_path)

    print(f"Saved logits to {output_dir}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Dataset and Dataloader
    # Using a simple transform for inference
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # We can generate for any split, let's use the test set
    dataset = CUBDataset(root_dir=DATA_DIR, split='test', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    for arch_str, encoders in ARCH_ENCODER_PAIRINGS.items():
        for encoder in encoders:
            model_name = f"{arch_str}_{encoder}"
            model_weights_path = os.path.join(BASE_MODELS_DIR, model_name, "outputs", "best_model.pth")

            if not os.path.exists(model_weights_path):
                print(f"WARNING: Weights not found for {model_name} at {model_weights_path}. Skipping.")
                continue

            ArchClass = ARCH_MAP[arch_str]
            model = ArchClass(
                encoder_name=encoder,
                encoder_weights=None, # Weights are loaded manually
                in_channels=3,
                classes=NUM_CLASSES,
            )
            
            model.load_state_dict(torch.load(model_weights_path, map_location=device))
            
            generate_logits_for_model(model, model_name, dataloader, device)

    print("\nLogit generation complete for all models.")

if __name__ == "__main__":
    main()

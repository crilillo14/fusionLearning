

def inference_from_paths(model, 
                         modelDir, 
                         test_dataloader, 
                         n=5, 
                         debug_viz=False) -> None:
    
    # limit to 50 samples
    """
    Samples n random datapoints from the test set using the underlying dataset's image paths,
    loads and visualizes both the true and predicted segmentation masks.
    """


    # Try to access the dataset object and its image paths
    dataset = test_dataloader.dataset
    if hasattr(dataset, 'dataset'):
        # This is a Subset, get the original dataset
        base_dataset = dataset.dataset
        indices = dataset.indices
        image_paths = [base_dataset.image_paths[i] for i in indices]
    else:
        image_paths = dataset.image_paths

    total_samples = len(image_paths)
    n = min(n, 50)                                          
    indices = random.sample(range(total_samples), n)


    model.to(device)
    model.load_state_dict(torch.load(modelDir + "outputs/best_model.pth", map_location=device))
    model.eval()
    
    fns = []
    plt.figure(figsize=(6, n * 3))
    for i, idx in enumerate(indices):
        

        # Get sample from dataset by index
        image, true_mask, filename = dataset[idx]
        fns.append(filename)
        image_input = image.to(device).unsqueeze(0)  # think indexing the test dataset doesnt apply transforms
        true_mask_np = true_mask.cpu().squeeze(0).numpy()

        with torch.no_grad():
            
            logits = model(image_input)

            if debug_viz:
                
                # --- quick sanity-check prints ---------------------------------
                print("logits shape:", logits.shape)
                print("dtype:", logits.dtype, "device:", logits.device)
                print("min/max:", logits.min().item(), logits.max().item())

                # view a tiny patch (top-left 5×5) to see actual numbers
                print("sample values:\n", logits[0, 0, :5, :5])
                # ---------------------------------------------------------------



            # CHANGE : following BCEwithLogits, using sigmoid for one class seg
            prob = torch.sigmoid(logits)
            pred_mask = prob.squeeze(1)
            # pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

            pred_mask_continuous = pred_mask.cpu().numpy().astype(np.float32)

        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(image.cpu().permute(1, 2, 0))
        plt.title(f"Sample {i+1}")
        plt.axis('off') 

        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(true_mask_np, cmap='gray')
        plt.title(f"True Mask")
        plt.axis('off') 

        plt.subplot(n, 3, i * 3 + 3)
        # Use continuous probabilities with colormap showing probability values
        im = plt.imshow(pred_mask_continuous[0], cmap='viridis', vmin=0, vmax=1)
        plt.title(f"Predicted Probability")
        plt.axis('off')
        # Add colorbar to show probability scale
        plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(modelDir + "outputs/inference.png", dpi=150, bbox_inches='tight')
    plt.show()

    plt.close()

    if debug_viz:
        pprint.pprint(fns)



def copy_best_model_to_weights(model_dir) -> None:
    """
    Copies the best model from the model's output directory to the weights directory.
    """

    src = os.path.join(model_dir, "outputs", "best_model.pth")
    dst = os.path.join(model_dir, "..", "..", "weights", modelName, f"{modelName}.pth")
    os.makedirs(dst, exist_ok = True)
    shutil.copy(src, dst)
    
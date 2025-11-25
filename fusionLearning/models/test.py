"""Run testing process"""

def test(modelDir, model, test_dataloader, lossFunc):
    model.eval()
    tloss = 0.0

    with torch.no_grad():
        for image, segmentation_mask, _ in tqdm(test_dataloader, desc=f"[TEST]", leave=False, ncols=80):
            image = image.to(device)
            segmentation_mask = segmentation_mask.to(device)

            logits = model(image)
            tloss += lossFunc(logits, segmentation_mask).item()

    avg_test_loss = tloss / len(test_dataloader)

    test_metrics = {
        'test_loss': avg_test_loss,
    }
    with open(modelDir + 'outputs/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f)

    print(f"Test: Loss={avg_test_loss:.4f}")
    return test_metrics
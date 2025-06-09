# preprocessing.md

Get data filenames
when next(dataloader) called,
    from fn, get img, mask as PIL
    convert both to torch.Tensor with uint8 dtype
    apply geometric transforms to img, mask
    apply photometric transforms to img
    return img, mask

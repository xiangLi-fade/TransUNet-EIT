import math
import os

import torch
from torch.utils.data import DataLoader

from datasets.dataset_eit import EITDataset


def circular_mask(size, device):
    axis = torch.linspace(-1, 1, size, device=device)
    y, x = torch.meshgrid(axis, axis)
    return (x.square() + y.square() <= 1).float()[None, None]


def masked_reconstruction_loss(prediction, target, mask):
    pixels = mask.sum() * prediction.size(0)
    diff = (prediction - target) * mask
    l1 = diff.abs().sum() / pixels
    mse = diff.square().sum() / pixels
    return 0.7 * l1 + 0.3 * mse, l1, mse


def evaluate(model, loader, device):
    model.eval()
    sums = {"mae": 0.0, "mse": 0.0}
    mask = circular_mask(128, device)
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            voltage = batch["voltage"].to(device)
            target = batch["conductivity"].to(device)
            _, mae, mse = masked_reconstruction_loss(model(image, voltage), target, mask)
            sums["mae"] += mae.item() * image.size(0)
            sums["mse"] += mse.item() * image.size(0)
    count = len(loader.dataset)
    return sums["mae"] / count, math.sqrt(sums["mse"] / count)


def train_eit(args, model, device):
    train_set = EITDataset(args.data_root, "train")
    val_set = EITDataset(args.data_root, "val")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers,
                              generator=generator, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    os.makedirs(args.output_dir, exist_ok=True)
    mask = circular_mask(128, device)
    best_mae = float("inf")
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_mae = checkpoint["best_mae"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"从 epoch {checkpoint['epoch']} 恢复训练")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            voltage = batch["voltage"].to(device)
            target = batch["conductivity"].to(device)
            loss, _, _ = masked_reconstruction_loss(model(image, voltage), target, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * image.size(0)
        scheduler.step()
        val_mae, val_rmse = evaluate(model, val_loader, device)
        print(f"epoch {epoch:03d} loss={running/len(train_set):.6f} val_mae={val_mae:.6f} val_rmse={val_rmse:.6f}")
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "best_mae": min(best_mae, val_mae), "args": vars(args)}
        torch.save(state, os.path.join(args.output_dir, "last.pth"))
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(state, os.path.join(args.output_dir, "best.pth"))

import argparse
import csv
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset_eit import EITDataset
from networks.eit_transunet import EITTransUNet
from trainer_eit import circular_mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EITTransUNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    dataset = EITDataset(args.data_root, "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    mask = circular_mask(128, device)
    maes, rmses, correlations, rows = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            target = batch["conductivity"].to(device)
            pred = model(batch["image"].to(device), batch["voltage"].to(device))
            valid = mask.bool().expand_as(pred)
            p, t = pred[valid], target[valid]
            maes.append((p - t).abs().mean().item())
            rmses.append(math.sqrt((p - t).square().mean().item()))
            p_centered, t_centered = p - p.mean(), t - t.mean()
            denominator = torch.sqrt(p_centered.square().sum() * t_centered.square().sum())
            correlations.append((p_centered * t_centered).sum().div(denominator + 1e-12).item())
            rows.append([batch["case_name"][0], maes[-1], rmses[-1], correlations[-1]])
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                np.save(os.path.join(args.save_dir, batch["case_name"][0] + "_pred.npy"), pred[0, 0].cpu().numpy())
    if args.metrics_csv:
        metrics_dir = os.path.dirname(os.path.abspath(args.metrics_csv))
        os.makedirs(metrics_dir, exist_ok=True)
        with open(args.metrics_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["case_name", "mae", "rmse", "cc"])
            writer.writerows(rows)
    print(f"MAE={np.mean(maes):.6f} RMSE={np.mean(rmses):.6f} CC={np.nanmean(correlations):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 EIT-TransUNet")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--metrics_csv", default="./predictions/metrics.csv")
    main(parser.parse_args())

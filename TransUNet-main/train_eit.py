import argparse
import random

import numpy as np
import torch

from networks.eit_transunet import EITTransUNet
from trainer_eit import train_eit


def parse_args():
    parser = argparse.ArgumentParser(description="训练 EIT-TransUNet")
    parser.add_argument("--data_root", required=True, help="含 train/ 和 val/ 的数据目录")
    parser.add_argument("--output_dir", default="./checkpoints/eit")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", default="", help="last.pth 路径，用于断点续训")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EITTransUNet().to(device)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    train_eit(args, model, device)

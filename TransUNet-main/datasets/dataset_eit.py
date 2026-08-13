from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EITDataset(Dataset):
    """读取包含 voltage、image、conductivity 的 EIT npz 数据。"""

    def __init__(self, root_dir, split="train"):
        root = Path(root_dir)
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"数据划分目录不存在: {split_dir}")
        search_dir = split_dir
        self.files = sorted(search_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"未在 {search_dir} 找到 .npz 文件")

        self.samples = []
        for path in self.files:
            with np.load(path) as data:
                self._validate(data, path)
                count = 1 if data["voltage"].ndim == 1 else data["voltage"].shape[0]
                self.samples.extend((path, i if count > 1 else None) for i in range(count))

    @staticmethod
    def _validate(data, path):
        required = {"voltage", "image", "conductivity"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} 缺少字段: {sorted(missing)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, sample_index = self.samples[index]
        with np.load(path) as data:
            voltage = data["voltage"] if sample_index is None else data["voltage"][sample_index]
            image = data["image"] if sample_index is None else data["image"][sample_index]
            target = data["conductivity"] if sample_index is None else data["conductivity"][sample_index]

        voltage = np.asarray(voltage, dtype=np.float32).reshape(-1)
        image = np.asarray(image, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if voltage.size != 208 or image.shape != (128, 128) or target.shape != (128, 128):
            raise ValueError(
                f"{path} 样本形状应为 (208,), (128,128), (128,128)，实际为 "
                f"{voltage.shape}, {image.shape}, {target.shape}"
            )

        # 输入按样本标准化；电导率真值保持原始物理尺度。
        voltage = (voltage - voltage.mean()) / (voltage.std() + 1e-6)
        image = (image - image.mean()) / (image.std() + 1e-6)
        return {
            "voltage": torch.from_numpy(voltage),
            "image": torch.from_numpy(image).unsqueeze(0),
            "conductivity": torch.from_numpy(target).unsqueeze(0),
            "case_name": f"{path.stem}_{sample_index}" if sample_index is not None else path.stem,
        }

# TransUNet-EIT

TransUNet-EIT 是一个用于二维电阻抗断层成像（Electrical Impedance Tomography, EIT）的双分支深度重建模型。模型同时使用相邻激励下的 208 维边界电压和传统算法生成的初始重建图，通过 Transformer 融合两种信息并输出连续电导率分布。

本项目基于 [TransUNet](https://github.com/Beckschen/TransUNet) 修改。原始医学图像分割代码仍保留在 `train.py`、`test.py` 和相关文件中。

## 模型结构

```text
边界电压 (208,) ── MLP ── 电压 tokens ─┐
                                        ├─ Transformer ─ Decoder ─ 电导率图 (128×128)
初始重建图 (128×128) ── Patch tokens ───┘
```

- 16 个电极，相邻激励，208 个有效电压测量值
- 初始重建图作为图像分支输入
- 电压 token 与图像 patch token 进行联合自注意力
- 仅解码图像 token，输出单通道连续电导率图
- L1 和 MSE 组合损失仅在圆形成像区域内计算

## 环境安装

推荐 Python 3.8 或更高版本。

```bash
pip install -r requirements-eit.txt
```

如需运行原始 TransUNet 医学分割代码，请安装 `requirements.txt` 中的额外依赖。

## 数据格式

数据目录必须包含独立的训练、验证和测试划分：

```text
data/eit/
├── train/
│   └── sample_0001.npz
├── val/
│   └── sample_0001.npz
└── test/
    └── sample_0001.npz
```

每个 `.npz` 文件包含：

```python
voltage       # float32, (208,)
image         # float32, (128, 128)，传统算法的初始重建图
conductivity  # float32, (128, 128)，目标电导率分布
```

也支持将一个划分保存为批量数组：`(N, 208)`、`(N, 128, 128)` 和 `(N, 128, 128)`。

电压和初始图在读取时进行逐样本标准化，目标电导率保持原始物理尺度。训练集、验证集和测试集应使用互不重叠的物理模型或样本。

## 快速验证

生成少量演示数据：

```bash
python scripts/generate_demo_data.py --output data/demo
```

演示数据仅用于检查代码流程，不是有限元 EIT 仿真数据，也不能用于评价真实重建性能。

执行训练：

```bash
python train_eit.py --data_root data/demo --epochs 100 --batch_size 8
```

从中断处恢复：

```bash
python train_eit.py --data_root data/demo --epochs 100 --resume checkpoints/eit/last.pth
```

测试并保存重建结果和逐样本指标：

```bash
python test_eit.py \
  --data_root data/demo \
  --checkpoint checkpoints/eit/best.pth \
  --save_dir predictions \
  --metrics_csv predictions/metrics.csv
```

测试报告包含：

- MAE：平均绝对误差
- RMSE：均方根误差
- CC：圆形成像域内的 Pearson 相关系数

## 主要文件

```text
datasets/dataset_eit.py       EIT 数据读取与校验
networks/eit_transunet.py     双分支融合网络
trainer_eit.py                圆域损失、训练和验证
train_eit.py                  训练入口
test_eit.py                   测试与指标导出
scripts/generate_demo_data.py 演示数据生成器
```

## 当前限制

- 输入维度固定为 208，图像尺寸固定为 128×128。
- 当前损失不包含有限元正演或电压一致性约束。
- 演示数据中的电压不是严格的 EIT 正演结果。
- 不同设备、激励协议或电极数量需要相应修改电压编码器和数据预处理。

## 引用

使用本项目时，请同时引用原始 TransUNet：

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L. and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```



## 许可证

本项目沿用原始项目的 [Apache License 2.0](LICENSE)。重新分发或修改时请保留许可证及原项目版权声明。

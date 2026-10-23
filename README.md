# MiVOLO 年龄与性别预测：ONNX / RK3588

基于 [WildChlamydia/MiVOLO](https://github.com/WildChlamydia/MiVOLO) 整理，提供 PyTorch 权重导出、RK3588 模型转换，以及图片批量测试。

原项目 Git 地址：`https://github.com/WildChlamydia/MiVOLO.git`。

当前导出为 **仅使用人脸的版本**：人体输入固定为空白。

## 目录说明
#models：从原项目中下载的 MiVOLO、YOLO 权重
#model_onnx：.pth 转换为 .onnx 后的模型
#model_rknn：.onnx 转换为 .rknn 后的模型
以上模型均可在 Releases V1.0.0 中进行下载
```text
Mivolo_age_gender/
├── README.md
├── .gitignore
└── MiVOLO/
    ├── models/                  # 下载的 MiVOLO、YOLO 权重
    └── mivolo/
        ├── model/               # 原始 MiVOLO 网络与模型加载代码
        ├── data/                # 图像预处理、数据读取等工具
        ├── input_images/        # 待测试图片
        ├── output_images/       # 标注图片及 CSV / JSON 结果
        ├── model_onnx/          # 导出的 ONNX 及同名 JSON
        ├── model_rknn/          # 转换后的 RKNN 及同名 JSON
        ├── export_onnx.py       # .pth.tar → 已简化的 ONNX
        ├── export_rknn.py       # ONNX → RK3588 RKNN
        ├── batch_test.py        # YOLO / OpenCV 检测 + ONNX 年龄性别预测
        ├── predictor.py         # 原项目 PyTorch 推理流程
        ├── structures.py       # 原项目检测结果、裁剪和关联结构
        └── requirements.txt     # 导出和测试所需的 Python 依赖
```

`input_images`、`output_images`、`model_onnx`、`model_rknn`、`models` 仅提交 `.gitkeep` 占位文件，图片、权重和生成结果不上传。

## 环境准备

以下项目命令均在 `MiVOLO` 目录执行。已有 `mivolo_env` 时直接激活即可；新环境可使用 Python 3.10：

```bash
cd MiVOLO
conda create -n mivolo_env python=3.10 -y
conda activate mivolo_env
python -m pip install -r mivolo/requirements.txt
```

注意：ONNX 导出和批量测试可在 Windows 或 Linux 中运行。
     onnx--->RKNN 转换需使用独立的 Linux / WSL2 环境及 RKNN-Toolkit2。在linux中使用当前的脚本 export_rknn.py 即可完成转换

## 下载权重

以下链接来自[原项目 README](https://github.com/WildChlamydia/MiVOLO#demo)，下载后放入 `MiVOLO/models/`：

| 用途 | 本地使用的文件名 | 下载地址 |
| --- | --- | --- |
| MiVOLO 年龄、性别模型 | `model_imdb_cross_person_4.22_99.46.pth.tar` | [Google Drive](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view) |
| YOLO 人脸、人体检测模型 | `yolov8x_person_face.pt` | [Google Drive](https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view) |

原项目将上述 MiVOLO 权重标注为 `4.24`，本地使用的下载文件名为 `4.22`。导出脚本可自动识别 `4.22`、`4.24` 和 `mivolo_imbd.pth.tar`；其他名称通过 `--checkpoint` 指定。

## 1. PyTorch 权重转 ONNX

```bash
conda activate mivolo_env
python mivolo/export_onnx.py
```

脚本自动加载权重，将 Fold 表达为 Gather/Add，执行 ONNX 简化及与 PyTorch 的数值比较，最后只保存：

```text
mivolo/model_onnx/mivolo_face_only_rknn.onnx
mivolo/model_onnx/mivolo_face_only_rknn.json
```

模型固定输入为 `[1, 3, 224, 224]`，无需再单独简化。指定其他权重时：

```bash
python mivolo/export_onnx.py --checkpoint models/你的权重.pth.tar
```

输出已存在时，加 `--overwrite` 重新导出。

## 2. ONNX 转 RKNN

在 Ubuntu / WSL2 中安装 [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)。本项目已使用 **Toolkit2 2.3.2、Linux x86_64、Python 3.8** 完成过转换；安装包可从[官方 packages 目录](https://github.com/airockchip/rknn-toolkit2/tree/v2.3.2/rknn-toolkit2/packages/x86_64)获取，wheel 必须匹配 Python 版本与系统架构。

新环境中，下载 `requirements_cp38-2.3.2.txt` 和对应的 `cp38` wheel 后，在它们所在目录安装：

```bash
conda create -n rknn_env python=3.8 -y
conda activate rknn_env
python -m pip install -r requirements_cp38-2.3.2.txt
python -m pip install rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

激活已安装 Toolkit2 的环境，在 Linux 中进入项目 `MiVOLO` 目录后执行：

```bash
conda activate rknn_env
python mivolo/export_rknn.py --check-only
python mivolo/export_rknn.py --output mivolo/model_rknn/mivolo_face_only_rk3588.rknn
```

此命令将 RKNN 和同名 JSON 写入 `model_rknn`；不传 `--output` 时，脚本默认写入 `models`。目标平台固定为 `rk3588`，当前关闭 INT8 量化，无需校准图片集。

如果单独复制到 WSL 的 `~/rknn_mivolo` 文件夹，需要同时复制 **`export_rknn.py`、ONNX、同名 JSON**，然后执行：

```bash
conda activate rknn_env
cd ~/rknn_mivolo
python export_rknn.py --onnx ./mivolo_face_only_rknn.onnx --output ./mivolo_face_only_rk3588.rknn
```

输出已存在时加 `--overwrite`。`--check-only` 只检查文件，实际转换需要运行不带该参数的命令。

## 3. 图片批量测试

将图片放入 `mivolo/input_images/`，在 `mivolo_env` 环境中运行以下任一种方式。

### 方式一：项目原 YOLO 检测器（默认）

```bash
python mivolo/batch_test.py --detector yolo
```

需要 `models/yolov8x_person_face.pt`。检测人脸后，使用 ONNX 预测年龄、性别；结果保存到 `mivolo/output_images/`。

### 方式二：OpenCV Haar 人脸检测器

```bash
python mivolo/batch_test.py --detector haar --output-dir mivolo/output_images/haar
```

不需要 YOLO 权重，仍使用同一份 ONNX 预测年龄、性别。单独输出到 `haar` 子目录，方便与 YOLO 对比。

两种方式均输出标注图片、`results.csv` 和 `results.json`，记录人脸框、年龄、性别、置信度与耗时。默认处理全部人脸，可加 `--largest-face` 只处理最大人脸。未检测到人脸时记录 `no_face`，不会自动把整张图片送入年龄模型。

## 使用注意

- **模型只接收裁剪后的人脸。** 先检测、裁剪，再等比例缩放并用黑边填充至 224×224；更换检测器会改变裁剪内容，可能影响预测结果。
- **ONNX 与 RKNN 的输入预处理不同：**

  | 模型 | 输入 | 归一化 |
  | --- | --- | --- |
  | ONNX | `float32`，NCHW，`[1,3,224,224]` | `(RGB / 255 - mean) / std` |
  | RKNN（普通 Python 推理接口） | `uint8`，NHWC，`[1,224,224,3]` | 已编译进模型，**不要再次归一化** |

- ONNX 的 `mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`。输出顺序为男性 logit、女性 logit、归一化年龄；性别对前两项做 softmax，年龄为 `output[2] * 94 + 48`。以配套 JSON 为准。
- 保留模型和同名 JSON，脚本会校验 SHA256；不要混用不同次导出的文件。当前版本的人体输入固定为空白，与原项目启用 `--with-persons` 的结果可能不同。
- `batch_test.py` 使用 ONNX Runtime CPU 推理；YOLO 方式的检测部分仍使用 PyTorch。RKNN 模型的板端推理需另接 RKNN-Toolkit-Lite2 / Runtime，并单独部署检测器。
- RKNN 编译成功后，仍需在 RK3588 上验证精度、驱动与 Runtime 兼容性，以及检测和年龄预测的总耗时。`check*.onnx` 是编译中间文件，可以删除。


# SAM2视频目标检测与追踪系统 - 使用手册

---

**版本**: v2.0  
**日期**: 2025年10月29日  
**适用平台**: Linux (Ubuntu)  
**Python版本**: ≥ 3.10  

---

## 📖 目录

1. [系统概述](#系统概述)
2. [环境要求](#环境要求)
3. [安装部署](#安装部署)
4. [目录结构](#目录结构)
5. [配置说明](#配置说明)
6. [运行方式](#运行方式)
7. [功能模块](#功能模块)
8. [故障排除](#故障排除)
9. [文件路径索引](#文件路径索引)
10. [技术支持](#技术支持)

---

## 🎯 系统概述

SAM2视频目标检测与追踪系统是一个基于最新SAM2和GroundingDINO技术的智能视频分析工具。系统具备以下核心能力：

### 主要功能
- **智能目标检测**: 基于自然语言描述检测视频中的任意目标
- **实时目标追踪**: 多目标连续追踪，保持ID一致性
- **孔洞检测专项**: 专门优化的孔洞检测与验证算法
- **实时摄像头**: 支持普通摄像头和RealSense深度摄像头
- **高质量分割**: 精确的目标分割和mask生成

### 技术特点
- **模块化架构**: 易于扩展和维护
- **多模式运行**: 视频文件处理、实时摄像头、Jupyter演示
- **智能验证**: 多重特征验证确保检测准确性
- **高性能**: GPU加速，支持批量处理

---

## 💻 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU（推荐RTX 3080及以上）
- **显存**: ≥ 8GB VRAM
- **内存**: ≥ 16GB RAM
- **存储**: ≥ 20GB 可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+)
- **Python**: ≥ 3.10
- **CUDA**: 12.1+
- **PyTorch**: ≥ 2.3.1
- **TorchVision**: ≥ 0.18.1

### 推荐环境
```bash
# 系统信息
OS: Ubuntu 20.04 LTS
Python: 3.10
CUDA: 12.1
PyTorch: 2.3.1
```

---

## 🚀 安装部署

### 步骤1: 环境准备

#### 1.1 创建Conda环境
```bash
# 创建专用环境
conda create -n sam2_env python=3.10 -y
conda activate sam2_env
```

#### 1.2 安装CUDA工具包
```bash
# 检查CUDA版本
nvcc --version

# 如果未安装，下载CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```

#### 1.3 设置环境变量
```bash
# 添加到 ~/.bashrc
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 重新加载环境变量
source ~/.bashrc
```

### 步骤2: 安装依赖包

#### 2.1 安装PyTorch
```bash
# 安装PyTorch (CUDA 12.1版本)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2.2 验证PyTorch安装
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"
```

#### 2.3 安装项目依赖
```bash
# 进入项目目录
cd /home/colab/anaconda3/envs/sam2_env_ozh/program/code

# 安装requirements.txt中的依赖
pip install -r requirements.txt
```

### 步骤3: 模型安装

#### 3.1 安装SAM2
```bash
# 进入任一SAM2目录
cd camera-new/Grounded-SAM-2

# 安装SAM2
pip install -e .

# 如果CUDA扩展构建失败，可以跳过CUDA扩展
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

#### 3.2 安装GroundingDINO
```bash
# 安装GroundingDINO
pip install --no-build-isolation -e grounding_dino
```

#### 3.3 下载预训练模型
```bash
# 下载SAM2模型
cd checkpoints
bash download_ckpts.sh

# 下载GroundingDINO模型
cd ../gdino_checkpoints
bash download_ckpts.sh
```

### 步骤4: 验证安装

#### 4.1 测试导入
```bash
python -c "
import torch
import sam2
import groundingdino
print('✓ 所有模块导入成功')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
"
```

#### 4.2 运行测试脚本
```bash
# 运行简单测试
cd sub_code
python test_sam2_import.py
```

---

## 📁 目录结构

```
/home/colab/anaconda3/envs/sam2_env_ozh/program/code/
├── requirements.txt                     # 依赖包列表
├── SAM2视频推理示例.ipynb              # Jupyter演示笔记本
│
├── input/                              # 输入文件目录
│   ├── images/                         # 图片输入
│   └── videos/                         # 视频输入
│
├── output/                             # 输出结果目录
│   ├── images/                         # 图片结果
│   ├── videos/                         # 视频结果
│   ├── videos_ffmpeg/                  # FFmpeg处理结果
│   └── videos_masks/                   # Mask数据
│
├── camera-new/                         # 实时摄像头模块
│   ├── 1camera.py                      # 摄像头主程序
│   ├── grounded_sam2_tracking_camera_with_continuous_id.py
│   ├── README_GroundingDINO_优化.md    # 优化说明
│   └── Grounded-SAM-2/                # SAM2核心代码
│
├── video-new/                          # 视频处理模块 (最新版本)
│
├── sub_code/                           # 子功能和工具
│   ├── picture_demo1.py                # 图片演示
│   ├── video_processor.py              # 视频处理器
│   ├── test_sam2_import.py             # 导入测试
│   ├── download_grounding_dino.py      # 模型下载
│   └── various_utility_scripts.py      # 各种工具脚本
│
└── 归档/                               # 历史版本归档
    ├── 7.30plus归档/                   # 7月30日版本 (稳定)
    │   ├── main.py                     # 主程序
    │   ├── run_detection.sh            # 一键运行脚本
    │   ├── README.md                   # 使用说明
    │   ├── modules/                    # 功能模块
    │   │   ├── config.py               # 配置管理
    │   │   ├── video_utils.py          # 视频工具
    │   │   ├── tracking_manager.py     # 追踪管理
    │   │   ├── hole_validator.py       # 孔洞验证
    │   │   ├── label_manager.py        # 标签管理
    │   │   └── overlap_manager.py      # 重叠管理
    │   └── Grounded-SAM-2/            # SAM2模型
    │
    ├── 归档7.30plus/                   # 7月30日增强版
    ├── 归档8.19video/                  # 8月19日视频版
    ├── 归档7.30cammer/                 # 7月30日摄像头版
    └── other_archived_versions/        # 其他历史版本
```

---

## ⚙️ 配置说明

### 主要配置文件

#### 1. requirements.txt (依赖配置)
```
路径: /home/colab/anaconda3/envs/sam2_env_ozh/program/code/requirements.txt
功能: 定义所有Python依赖包及版本
重要包: torch>=2.3.1, transformers>=4.20.0, opencv-python>=4.5.0
```

#### 2. config.py (系统配置)
```
路径: 归档/7.30plus归档/modules/config.py
功能: 系统参数配置中心
```

**主要配置参数:**
```python
# 检测参数
detection_params = {
    'text_prompt': 'hole',           # 检测目标描述
    'box_threshold': 0.31314,        # 边框置信度阈值
    'text_threshold': 0.21314,       # 文本匹配阈值
    'step': 20,                      # 检测间隔帧数
}

# 孔洞验证参数
hole_validation = {
    'min_area_ratio': 0.001,         # 最小面积比例 (0.1%)
    'max_area_ratio': 0.15,          # 最大面积比例 (15%)
    'brightness_threshold': 0.7,     # 亮度阈值
    'contrast_threshold': 0.3,       # 对比度阈值
    'texture_threshold': 0.4,        # 纹理阈值
}

# 追踪参数
tracking_params = {
    'max_movement_ratio': 0.2,       # 最大移动比例 (20%)
    'max_size_change_ratio': 0.6,    # 最大尺寸变化 (60%)
    'max_missing_frames': 30,        # 最大缺失帧数
}
```

#### 3. 模型路径配置
```python
# SAM2 模型路径
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
sam2_model_cfg = "sam2_hiera_l.yaml"

# GroundingDINO 模型路径  
grounding_dino_checkpoint = "./gdino_checkpoints/groundingdino_swint_ogc.pth"
grounding_dino_config = "./grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
```

---

## 🚀 运行方式

### 方式1: 孔洞检测系统 (推荐)

#### 一键启动
```bash
# 进入稳定版本目录
cd 归档/7.30plus归档

# 准备输入视频
cp your_video.mp4 ../../input/videos/

# 运行一键启动脚本
chmod +x run_detection.sh
./run_detection.sh
```

#### 手动运行
```bash
# 进入目录
cd 归档/7.30plus归档

# 直接运行主程序
python main.py
```

**输出文件:**
- `../../output/videos/hole_detection_result_XXfps.mp4` - 检测结果视频
- `../../output/mask_data/mask_frameXXX.npy` - 分割mask数据
- `../../output/json_data/frameXXX.json` - 检测数据JSON

### 方式2: 实时摄像头检测

#### 普通摄像头
```bash
cd camera-new
python 1camera.py
```

#### RealSense深度摄像头
```bash
cd camera-new
python 1camera.py --camera realsense
```

#### 指定检测目标
```bash
# 检测人员
python 1camera.py --target "person"

# 检测车辆
python 1camera.py --target "car"

# 高精度模式
python 1camera.py --model accurate --target "hole"
```

### 方式3: Jupyter演示

```bash
# 启动Jupyter
jupyter notebook

# 打开演示文件
# 浏览器中打开: SAM2视频推理示例.ipynb
```

### 方式4: 图片处理

```bash
cd sub_code

# 处理单张图片
python picture_demo1.py --input ../input/images/test.jpg

# 批量处理图片
python picture_demo1.py --input ../input/images/ --batch
```

### 方式5: 视频批量处理

```bash
cd sub_code

# 批量处理视频
python video_processor.py --input ../input/videos/ --output ../output/videos/
```

---

## 🧩 功能模块

### 1. 核心检测模块

#### video_utils.py
```
路径: 归档/7.30plus归档/modules/video_utils.py
功能: 视频读取、写入、帧处理
主要函数:
- create_video_reader() - 创建视频读取器
- create_video_writer() - 创建视频写入器
- extract_frames() - 提取视频帧
```

#### tracking_manager.py
```
路径: 归档/7.30plus归档/modules/tracking_manager.py
功能: 多目标追踪管理
主要功能:
- ID一致性维护
- 运动预测
- 目标关联
- 历史轨迹管理
```

#### hole_validator.py
```
路径: 归档/7.30plus归档/modules/hole_validator.py
功能: 孔洞特征验证
验证维度:
- 面积比例验证
- 亮度对比验证
- 纹理特征验证
- 形状特征验证
- 边缘特征验证
```

### 2. 工具模块

#### label_manager.py
```
功能: 标签和注释管理
- 目标标签分配
- 颜色管理
- 文本渲染
```

#### overlap_manager.py
```
功能: 重叠检测和处理
- 目标重叠检测
- NMS非极大值抑制
- 重叠区域处理
```

### 3. 摄像头模块

#### grounded_sam2_tracking_camera_with_continuous_id.py
```
路径: camera-new/grounded_sam2_tracking_camera_with_continuous_id.py
功能: 实时摄像头检测和追踪
特点:
- 支持普通摄像头和RealSense
- 实时目标检测
- 连续ID追踪
- 可视化显示
```

### 4. 测试和工具

#### test_sam2_import.py
```
路径: sub_code/test_sam2_import.py
功能: 验证SAM2安装和导入
```

#### download_grounding_dino.py
```
路径: sub_code/download_grounding_dino.py
功能: 下载GroundingDINO模型
```

---

## 🛠️ 故障排除

### 常见问题及解决方案

#### 1. 模型导入错误

**问题**: `ImportError: cannot import name '_C' from 'sam2'`

**解决方案**:
```bash
# 重新安装SAM2
cd Grounded-SAM-2
pip uninstall -y SAM-2
rm -f ./sam2/*.so
pip install -e ".[notebooks]"

# 如果仍然失败，跳过CUDA扩展
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

#### 2. CUDA环境问题

**问题**: `CUDA_HOME environment variable is not set`

**解决方案**:
```bash
# 设置CUDA环境变量
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 验证CUDA设置
python -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)"
```

#### 3. 内存不足

**问题**: `CUDA out of memory`

**解决方案**:
```bash
# 设置内存分配策略
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 在代码中调整参数
# 增加step值，减少检测频率
# 使用较小的SAM2模型
# 减小输入视频分辨率
```

#### 4. 模型文件缺失

**问题**: 找不到模型文件

**解决方案**:
```bash
# 检查模型文件路径
ls checkpoints/sam2_hiera_large.pt
ls gdino_checkpoints/groundingdino_swint_ogc.pth

# 如果缺失，重新下载
cd checkpoints && bash download_ckpts.sh
cd gdino_checkpoints && bash download_ckpts.sh
```

#### 5. 摄像头连接问题

**问题**: 摄像头无法打开

**解决方案**:
```bash
# 检查摄像头设备
ls /dev/video*

# 测试摄像头
v4l2-ctl --list-devices

# 检查权限
sudo usermod -a -G video $USER
```

#### 6. FFmpeg相关错误

**问题**: 视频编解码错误

**解决方案**:
```bash
# 安装完整的FFmpeg
sudo apt update
sudo apt install ffmpeg

# 检查FFmpeg版本
ffmpeg -version
```

### 性能优化建议

#### 1. GPU优化
```python
# 使用混合精度
torch.backends.cudnn.benchmark = True

# 优化显存使用
torch.cuda.empty_cache()
```

#### 2. 处理优化
```python
# 调整检测间隔
detection_params['step'] = 30  # 每30帧检测一次

# 降低分辨率
input_size = (640, 480)  # 而不是(1280, 720)
```

#### 3. 批处理优化
```python
# 批量处理多个视频
batch_size = 4
process_videos_in_batch(video_list, batch_size)
```

---

## 📋 文件路径索引

### 主要程序文件

| 文件名 | 路径 | 功能描述 |
|--------|------|----------|
| main.py | 归档/7.30plus归档/main.py | 孔洞检测主程序 |
| 1camera.py | camera-new/1camera.py | 实时摄像头检测 |
| SAM2视频推理示例.ipynb | SAM2视频推理示例.ipynb | Jupyter演示笔记本 |
| picture_demo1.py | sub_code/picture_demo1.py | 图片检测演示 |

### 配置和工具文件

| 文件名 | 路径 | 功能描述 |
|--------|------|----------|
| requirements.txt | requirements.txt | Python依赖列表 |
| config.py | 归档/7.30plus归档/modules/config.py | 系统配置 |
| run_detection.sh | 归档/7.30plus归档/run_detection.sh | 一键启动脚本 |

### 核心模块文件

| 模块名 | 路径 | 功能描述 |
|--------|------|----------|
| video_utils.py | 归档/7.30plus归档/modules/video_utils.py | 视频处理工具 |
| tracking_manager.py | 归档/7.30plus归档/modules/tracking_manager.py | 追踪管理 |
| hole_validator.py | 归档/7.30plus归档/modules/hole_validator.py | 孔洞验证 |
| label_manager.py | 归档/7.30plus归档/modules/label_manager.py | 标签管理 |
| overlap_manager.py | 归档/7.30plus归档/modules/overlap_manager.py | 重叠管理 |

### 测试和工具文件

| 文件名 | 路径 | 功能描述 |
|--------|------|----------|
| test_sam2_import.py | sub_code/test_sam2_import.py | SAM2导入测试 |
| download_grounding_dino.py | sub_code/download_grounding_dino.py | 模型下载工具 |
| video_processor.py | sub_code/video_processor.py | 视频批处理器 |

### 模型和数据目录

| 目录名 | 路径 | 内容描述 |
|--------|------|----------|
| input/ | input/ | 输入文件（图片/视频） |
| output/ | output/ | 输出结果 |
| checkpoints/ | */checkpoints/ | SAM2模型文件 |
| gdino_checkpoints/ | */gdino_checkpoints/ | GroundingDINO模型文件 |

### SAM2核心代码

| 目录名 | 路径 | 内容描述 |
|--------|------|----------|
| Grounded-SAM-2/ | camera-new/Grounded-SAM-2/ | 最新SAM2代码 |
| Grounded-SAM-2/ | 归档/7.30plus归档/Grounded-SAM-2/ | 稳定版SAM2代码 |

---

## 📞 技术支持

### 常用命令

#### 环境检查
```bash
# 检查Python环境
python --version
which python

# 检查CUDA环境  
nvcc --version
nvidia-smi

# 检查PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### 模型验证
```bash
# 检查SAM2
python -c "import sam2; print('SAM2导入成功')"

# 检查GroundingDINO
python -c "import groundingdino; print('GroundingDINO导入成功')"
```

#### 日志调试
```bash
# 启用详细日志
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# 运行程序
python main.py --verbose
```

### 联系信息

- **项目路径**: `/home/colab/anaconda3/envs/sam2_env_ozh/program/code/`
- **环境名称**: `sam2_env_ozh`
- **主要版本**: 归档/7.30plus归档/ (稳定版)
- **最新版本**: camera-new/ (实时版)

### 快速诊断清单

1. **Python环境**: ✓ Python ≥ 3.10
2. **CUDA驱动**: ✓ CUDA 12.1+
3. **PyTorch**: ✓ PyTorch ≥ 2.3.1
4. **模型文件**: ✓ SAM2 + GroundingDINO 模型
5. **依赖包**: ✓ requirements.txt 全部安装
6. **权限设置**: ✓ 摄像头和文件读写权限

---

**🎉 祝您使用愉快！如有问题，请参考故障排除部分或联系技术支持。**

---

*最后更新: 2025年10月29日*

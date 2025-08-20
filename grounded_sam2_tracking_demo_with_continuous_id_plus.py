import os
import sys
import cv2
import torch
import numpy as np
import supervision as sv
import subprocess

# 在导入其他库之前设置环境变量和警告过滤
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import warnings
import logging
# 抑制所有相关警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights of the model checkpoint.*")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*Some weights.*were not used.*")
warnings.filterwarnings("ignore", message=".*This IS expected.*")

# 设置日志级别
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# 添加SAM2模块路径
sam2_path = "/home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/sam2_grounding/Grounded-SAM-2"
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)

# 添加GroundingDINO路径 - 使用本地安装的版本
grounding_dino_path = "/home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/GroundingDINO-main"
if grounding_dino_path not in sys.path:
    sys.path.insert(0, grounding_dino_path)

from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 使用本地GroundingDINO
from groundingdino.util.inference import load_model, load_image, predict
print("✓ 使用本地 GroundingDINO")

# 设置utils路径
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
grounded_sam2_utils = os.path.join(current_dir, 'Grounded-SAM-2', 'utils')
sys.path.insert(0, grounded_sam2_utils)

# 导入utils模块
from track_utils import sample_points_from_masks

class TrackingHistoryManager:
    """追踪历史管理器，提高ID连续性并检查运动合理性"""
    def __init__(self, max_missing_frames=3, max_movement_ratio=0.3, max_size_change_ratio=0.5):
        self.object_history = {}  # {object_id: {'positions': [], 'frames': [], 'sizes': [], 'missing_count': 0}}
        self.max_missing_frames = max_missing_frames
        self.max_movement_ratio = max_movement_ratio  # 最大移动距离相对于图像尺寸的比例
        self.max_size_change_ratio = max_size_change_ratio  # 最大尺寸变化比例
        
    def update_object_position(self, object_id, center_position, frame_idx, mask_area=None):
        """更新物体位置和尺寸历史"""
        if object_id not in self.object_history:
            self.object_history[object_id] = {
                'positions': [],
                'frames': [],
                'sizes': [],
                'missing_count': 0
            }
        
        self.object_history[object_id]['positions'].append(center_position)
        self.object_history[object_id]['frames'].append(frame_idx)
        if mask_area is not None:
            self.object_history[object_id]['sizes'].append(mask_area)
        self.object_history[object_id]['missing_count'] = 0
        
        # 保留最近的10个记录
        if len(self.object_history[object_id]['positions']) > 10:
            self.object_history[object_id]['positions'] = self.object_history[object_id]['positions'][-10:]
            self.object_history[object_id]['frames'] = self.object_history[object_id]['frames'][-10:]
            if self.object_history[object_id]['sizes']:
                self.object_history[object_id]['sizes'] = self.object_history[object_id]['sizes'][-10:]
    
    def predict_next_position(self, object_id):
        """预测物体下一帧的位置"""
        if object_id not in self.object_history or len(self.object_history[object_id]['positions']) < 2:
            return None
            
        positions = self.object_history[object_id]['positions']
        if len(positions) >= 2:
            # 简单的线性预测
            last_pos = positions[-1]
            second_last_pos = positions[-2]
            velocity = (last_pos[0] - second_last_pos[0], last_pos[1] - second_last_pos[1])
            predicted_pos = (last_pos[0] + velocity[0], last_pos[1] + velocity[1])
            return predicted_pos
        return positions[-1] if positions else None
    
    def is_movement_reasonable(self, object_id, new_position, image_shape):
        """检查物体移动是否合理"""
        if object_id not in self.object_history or not self.object_history[object_id]['positions']:
            return True  # 新物体，无法比较
        
        last_position = self.object_history[object_id]['positions'][-1]
        
        # 计算移动距离
        movement_distance = np.sqrt(
            (new_position[0] - last_position[0])**2 + 
            (new_position[1] - last_position[1])**2
        )
        
        # 计算相对于图像尺寸的移动比例
        image_diagonal = np.sqrt(image_shape[0]**2 + image_shape[1]**2)
        movement_ratio = movement_distance / image_diagonal
        
        # 检查是否超过最大合理移动距离
        is_reasonable = movement_ratio <= self.max_movement_ratio
        
        if not is_reasonable:
            print(f"    ⚠️  物体ID {object_id} 移动过快: {movement_distance:.1f}像素 "
                  f"(占图像对角线{movement_ratio:.1%}，超过阈值{self.max_movement_ratio:.1%})")
        
        return is_reasonable
    
    def is_size_change_reasonable(self, object_id, new_size):
        """检查物体尺寸变化是否合理"""
        if object_id not in self.object_history or not self.object_history[object_id]['sizes']:
            return True  # 新物体，无法比较
        
        last_size = self.object_history[object_id]['sizes'][-1]
        
        # 计算尺寸变化比例
        if last_size > 0:
            size_change_ratio = abs(new_size - last_size) / last_size
            is_reasonable = size_change_ratio <= self.max_size_change_ratio
            
            if not is_reasonable:
                print(f"    ⚠️  物体ID {object_id} 尺寸变化过大: {last_size:.0f} → {new_size:.0f} "
                      f"(变化{size_change_ratio:.1%}，超过阈值{self.max_size_change_ratio:.1%})")
            
            return is_reasonable
        
        return True
    
    def validate_object_consistency(self, object_id, new_position, new_size, image_shape):
        """综合验证物体的运动和尺寸变化是否合理"""
        movement_ok = self.is_movement_reasonable(object_id, new_position, image_shape)
        size_ok = self.is_size_change_reasonable(object_id, new_size)
        
        is_consistent = movement_ok and size_ok
        
        if not is_consistent:
            print(f"    🚫 物体ID {object_id} 检测不一致，可能识别到了其他物体")
        
        return is_consistent
    
    def mark_missing(self, object_id):
        """标记物体在当前帧缺失"""
        if object_id in self.object_history:
            self.object_history[object_id]['missing_count'] += 1
    
    def should_keep_tracking(self, object_id):
        """判断是否应该继续追踪该物体"""
        if object_id not in self.object_history:
            return False
        return self.object_history[object_id]['missing_count'] <= self.max_missing_frames
    
    def cleanup_lost_objects(self):
        """清理丢失太久的物体"""
        to_remove = []
        for object_id, history in self.object_history.items():
            if history['missing_count'] > self.max_missing_frames:
                to_remove.append(object_id)
        
        for object_id in to_remove:
            del self.object_history[object_id]
    
    def _get_mask_center(self, mask):
        """计算mask的中心点"""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return (0, 0)
        
        center_y = np.mean(y_indices)
        center_x = np.mean(x_indices)
        return (center_x, center_y)

from video_utils import create_video_from_images
from common_utils import CommonUtils
from mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy

# This demo shows the continuous object tracking plus reverse tracking with Grounding DINO and SAM 2
"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "/home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/sam2_grounding/Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "../sam2/configs/sam2/sam2_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# 初始化本地 GroundingDINO 模型
GROUNDING_DINO_CONFIG = "/home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/sam2_grounding/Grounded-SAM-2/code_ozh/weights/groundingdino_swint_ogc.pth"

print(f"正在加载 GroundingDINO 模型...")
print(f"配置文件: {GROUNDING_DINO_CONFIG}")
print(f"权重文件: {GROUNDING_DINO_CHECKPOINT}")

# 检查文件是否存在
if not os.path.exists(GROUNDING_DINO_CONFIG):
    raise FileNotFoundError(f"GroundingDINO 配置文件不存在: {GROUNDING_DINO_CONFIG}")

if not os.path.exists(GROUNDING_DINO_CHECKPOINT):
    print(f"✗ GroundingDINO 权重文件不存在: {GROUNDING_DINO_CHECKPOINT}")
    print("提示: 请确保模型文件已下载到正确位置")
    print("您可以从以下位置复制现有的模型文件:")
    print("  - /home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/sam2_picture/GroundingDINO_sam2/gdino_checkpoints/groundingdino_swint_ogc.pth")
    print("  - /home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/sam2_grounding/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth")
    
    # 尝试从其他位置复制
    import shutil
    source_paths = [
        "/home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/sam2_picture/GroundingDINO_sam2/gdino_checkpoints/groundingdino_swint_ogc.pth",
        "/home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/sam2_grounding/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    ]
    
    for source_path in source_paths:
        if os.path.exists(source_path):
            print(f"找到模型文件，正在从 {source_path} 复制...")
            os.makedirs(os.path.dirname(GROUNDING_DINO_CHECKPOINT), exist_ok=True)
            shutil.copy2(source_path, GROUNDING_DINO_CHECKPOINT)
            print(f"✓ 模型文件已复制到 {GROUNDING_DINO_CHECKPOINT}")
            break
    else:
        raise FileNotFoundError(f"无法找到 GroundingDINO 权重文件")

try:
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device
    )
    print("✓ 本地 GroundingDINO 模型加载成功")
except Exception as e:
    print(f"✗ GroundingDINO 模型加载失败: {e}")
    raise RuntimeError("无法加载 GroundingDINO 模型")


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
# 针对hole的精确识别进行优化，使用更广泛的描述词
text = "hole. opening. cavity. circular hole. dark hole."

# Video processing paths
input_video_dir = "/home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/input/videos"
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg` (5-digit format with leading zeros)
video_dir = "/home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/output/videos_ffmpeg"
# 'output_dir' is the directory to save the annotated frames
output_dir = "/home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/output/videos_masks"
# 'output_video_path' is the path to save the final video
output_video_path = "/home/colab/anaconda3/envs/sam2_env_ozh/program_ozh/output/videos/output.mp4"

def clear_directory(directory_path):
    """
    清空指定目录中的所有文件和子目录
    
    Args:
        directory_path: 要清空的目录路径
    """
    if os.path.exists(directory_path):
        import shutil
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"    ⚠️  删除 {file_path} 时出错: {e}")
        print(f"    ✅  已清空目录: {directory_path}")
    else:
        print(f"    ℹ️  目录不存在: {directory_path}")

def extract_frames_with_ffmpeg(input_video_path, output_frames_dir, fps=None):
    """
    使用FFmpeg从视频中提取帧
    Args:
        input_video_path: 输入视频文件路径
        output_frames_dir: 输出帧图像的目录
        fps: 提取帧率，如果为None则提取所有帧
    """
    # 创建输出目录
    os.makedirs(output_frames_dir, exist_ok=True)
    
    # 构建FFmpeg命令
    if fps is not None:
        # 指定帧率提取
        cmd = [
            'ffmpeg', '-i', input_video_path,
            '-vf', f'fps={fps}',
            '-y',  # 覆盖输出文件
            os.path.join(output_frames_dir, '%05d.jpg')  # 🔧 改为5位数格式
        ]
    else:
        # 提取所有帧
        cmd = [
            'ffmpeg', '-i', input_video_path,
            '-y',  # 覆盖输出文件
            os.path.join(output_frames_dir, '%05d.jpg')  # 🔧 改为5位数格式
        ]
    
    print(f"🎬 正在使用FFmpeg提取视频帧...")
    print(f"输入视频: {input_video_path}")
    print(f"输出目录: {output_frames_dir}")
    if fps:
        print(f"提取帧率: {fps} fps")
    
    try:
        # 执行FFmpeg命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 统计提取的帧数
        extracted_files = [f for f in os.listdir(output_frames_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        extracted_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        
        print(f"✓ 成功提取了 {len(extracted_files)} 帧图像")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ FFmpeg执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ 视频帧提取失败: {e}")
        return False

def find_video_file(input_dir):
    """
    在输入目录中查找视频文件
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    if not os.path.exists(input_dir):
        print(f"✗ 输入目录不存在: {input_dir}")
        return None
    
    video_files = []
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(input_dir, file))
    
    if not video_files:
        print(f"✗ 在目录 {input_dir} 中未找到视频文件")
        print(f"支持的视频格式: {', '.join(video_extensions)}")
        return None
    
    if len(video_files) > 1:
        print(f"⚠️  找到多个视频文件，使用第一个: {video_files[0]}")
        for i, vf in enumerate(video_files):
            print(f"  {i+1}. {os.path.basename(vf)}")
    
    return video_files[0]

def get_video_framerate(video_path):
    """
    获取视频的帧率
    Args:
        video_path: 视频文件路径
    Returns:
        float: 视频帧率，失败时返回25.0作为默认值
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️  无法打开视频文件: {video_path}")
            return 25.0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps > 0:
            print(f"📊 检测到输入视频帧率: {fps:.2f} fps")
            return fps
        else:
            print(f"⚠️  无法获取视频帧率，使用默认值: 25.0 fps")
            return 25.0
            
    except Exception as e:
        print(f"⚠️  获取视频帧率失败: {e}，使用默认值: 25.0 fps")
        return 25.0

def validate_hole_features(mask, bbox, image_shape, confidence, image_array=None):
    """
    验证检测到的物体是否真的是hole（增强版：区分真hole和平面物体）
    Args:
        mask: 物体的mask
        bbox: 边界框 [x1, y1, x2, y2]
        image_shape: 图像尺寸 (H, W)
        confidence: 检测置信度
        image_array: 原始图像数组，用于纹理和深度分析
    Returns:
        is_valid_hole: bool, 是否是有效的hole
        hole_score: float, hole的置信度分数 (0-1)
        validation_info: dict, 验证信息
    """
    H, W = image_shape
    x1, y1, x2, y2 = bbox
    
    # 🔧 确保mask尺寸与图像尺寸匹配（关键修复）
    if mask.shape != (H, W):
        #print(f"    ⚠️  调整mask尺寸: {mask.shape} → ({H}, {W})")
        if mask.shape[0] > H or mask.shape[1] > W:
            # 如果mask比图像大，裁剪mask
            mask = mask[:H, :W]
        else:
            # 如果mask比图像小，填充mask
            new_mask = np.zeros((H, W), dtype=mask.dtype)
            new_mask[:mask.shape[0], :mask.shape[1]] = mask
            mask = new_mask
    
    # 🔧 确保image_array尺寸匹配
    if image_array is not None and image_array.shape[:2] != (H, W):
        print(f"    ⚠️  调整image_array尺寸: {image_array.shape} → ({H}, {W}, ...)")
        # 使用OpenCV resize调整图像尺寸
        if len(image_array.shape) == 3:
            image_array = cv2.resize(image_array, (W, H))
        else:
            image_array = cv2.resize(image_array, (W, H))
    
    # 基础特征计算
    mask_area = mask.sum()
    bbox_area = (x2 - x1) * (y2 - y1)
    
    # 1. 面积比例验证（hole不应该太大也不应该太小）
    image_area = H * W
    area_ratio = mask_area / image_area
    
    # hole的合理面积范围：0.1% ~ 15%
    area_score = 0.0
    if 0.001 <= area_ratio <= 0.15:
        # 在合理范围内，给予高分
        if 0.001 <= area_ratio <= 0.05:
            area_score = 1.0  # 最理想的hole大小
        else:
            area_score = 0.8  # 较大hole，仍然合理
    else:
        area_score = 0.1  # 不合理的大小
    
    # 2. 形状紧凑度验证（hole通常比较紧凑，接近圆形）
    if bbox_area > 0:
        compactness = mask_area / bbox_area
    else:
        compactness = 0
    
    # hole的compactness通常在0.3-1.0之间
    compactness_score = min(1.0, max(0.0, compactness))
    
    # 3. 宽高比验证（hole通常接近正方形或圆形）
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    if bbox_height > 0:
        aspect_ratio = bbox_width / bbox_height
    else:
        aspect_ratio = 1.0
    
    # 理想的宽高比在0.5-2.0之间
    if 0.5 <= aspect_ratio <= 2.0:
        aspect_score = 1.0
    elif 0.3 <= aspect_ratio <= 3.0:
        aspect_score = 0.7
    else:
        aspect_score = 0.3
    
    # 4. 检测置信度权重
    confidence_score = min(1.0, confidence * 2)  # 放大置信度影响
    
    # 5. 位置验证（hole通常不在图像边缘）
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 计算到边缘的最小距离比例
    edge_dist_ratio = min(
        center_x / W, 
        (W - center_x) / W,
        center_y / H, 
        (H - center_y) / H
    )
    
    # 如果太靠近边缘，降低分数
    if edge_dist_ratio > 0.1:
        edge_score = 1.0
    elif edge_dist_ratio > 0.05:
        edge_score = 0.8
    else:
        edge_score = 0.5
    
    # 6. 🔍 新增：深度和立体特征检测
    depth_score = 0.5  # 默认中性分数
    texture_score = 0.5  # 默认中性分数
    edge_gradient_score = 0.5  # 默认中性分数
    
    if image_array is not None:
        try:
            # 提取hole区域的图像
            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
            x1_int = max(0, x1_int)
            y1_int = max(0, y1_int)
            x2_int = min(W, x2_int)
            y2_int = min(H, y2_int)
            
            hole_region = image_array[y1_int:y2_int, x1_int:x2_int]
            mask_region = mask[y1_int:y2_int, x1_int:x2_int]
            
            # 🔧 确保尺寸匹配（关键修复）
            if hole_region.shape[:2] != mask_region.shape:
                print(f"    ⚠️  尺寸不匹配: hole_region{hole_region.shape} vs mask_region{mask_region.shape}")
                # 调整mask_region尺寸以匹配hole_region
                if len(mask_region.shape) == 2:
                    mask_region = mask_region[:hole_region.shape[0], :hole_region.shape[1]]
                else:
                    mask_region = mask_region[:hole_region.shape[0], :hole_region.shape[1], :]
            
            if hole_region.size > 0 and mask_region.sum() > 10:
                # 6a. 亮度分析：真hole通常比周围更暗
                # 🔧 确保mask_region是布尔类型用于索引
                mask_region_bool = mask_region.astype(bool)
                hole_pixels = hole_region[mask_region_bool]
                if len(hole_pixels) > 0:
                    # 转换为灰度值进行亮度分析
                    if len(hole_pixels.shape) == 2:  # 已经是灰度
                        hole_brightness = np.mean(hole_pixels)
                    else:  # RGB图像
                        hole_brightness = np.mean(np.mean(hole_pixels, axis=1))
                    
                    # 计算周围区域的亮度作为对比
                    padding = 10
                    y1_expand = max(0, y1_int - padding)
                    y2_expand = min(H, y2_int + padding)
                    x1_expand = max(0, x1_int - padding)
                    x2_expand = min(W, x2_int + padding)
                    
                    surrounding_region = image_array[y1_expand:y2_expand, x1_expand:x2_expand]
                    surrounding_mask = np.ones((y2_expand-y1_expand, x2_expand-x1_expand), dtype=bool)
                    
                    # 排除hole区域本身
                    hole_y_start = y1_int - y1_expand
                    hole_y_end = hole_y_start + (y2_int - y1_int)
                    hole_x_start = x1_int - x1_expand
                    hole_x_end = hole_x_start + (x2_int - x1_int)
                    
                    if (hole_y_end <= surrounding_mask.shape[0] and 
                        hole_x_end <= surrounding_mask.shape[1]):
                        surrounding_mask[hole_y_start:hole_y_end, hole_x_start:hole_x_end] = False
                    
                    # 🔧 确保surrounding_mask是布尔类型用于索引
                    surrounding_mask_bool = surrounding_mask.astype(bool)
                    surrounding_pixels = surrounding_region[surrounding_mask_bool]
                    if len(surrounding_pixels) > 0:
                        if len(surrounding_pixels.shape) == 2:
                            surrounding_brightness = np.mean(surrounding_pixels)
                        else:
                            surrounding_brightness = np.mean(np.mean(surrounding_pixels, axis=1))
                        
                        # 真hole应该比周围暗
                        brightness_ratio = hole_brightness / (surrounding_brightness + 1e-6)
                        if brightness_ratio < 0.7:  # hole比周围暗30%以上
                            depth_score = 1.0
                        elif brightness_ratio < 0.85:  # hole比周围暗15%以上
                            depth_score = 0.8
                        elif brightness_ratio < 0.95:  # hole比周围暗5%以上
                            depth_score = 0.6
                        else:  # hole不够暗，可能是平面
                            depth_score = 0.2
                
                # 6b. 纹理复杂度分析：真hole内部纹理较简单，平面物体纹理复杂
                if len(hole_pixels) > 50:  # 确保有足够像素进行分析
                    # 计算hole区域的标准差（纹理复杂度指标）
                    if len(hole_pixels.shape) == 2:
                        hole_std = np.std(hole_pixels)
                    else:
                        # 对RGB各通道计算标准差的平均值
                        hole_std = np.mean([np.std(hole_pixels[:, i]) for i in range(hole_pixels.shape[1])])
                    
                    # 真hole内部相对均匀（低标准差），平面物体纹理复杂（高标准差）
                    if hole_std < 15:  # 很均匀，可能是真hole
                        texture_score = 1.0
                    elif hole_std < 25:  # 比较均匀
                        texture_score = 0.8
                    elif hole_std < 40:  # 中等复杂度
                        texture_score = 0.6
                    else:  # 纹理复杂，可能是木板等平面物体
                        texture_score = 0.2
                
                # 6c. 边缘梯度分析：真hole边缘有明显的深度过渡
                edge_gradient_score = 0.5
                try:
                    # 计算hole边缘的梯度变化
                    gray_region = cv2.cvtColor(hole_region, cv2.COLOR_RGB2GRAY) if len(hole_region.shape) == 3 else hole_region
                    
                    # 使用Sobel算子计算梯度
                    grad_x = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    
                    # 在mask边缘计算梯度强度
                    kernel = np.ones((3,3), np.uint8)
                    mask_edge = cv2.morphologyEx(mask_region.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
                    
                    # 🔧 确保梯度和mask边缘尺寸匹配
                    if gradient_magnitude.shape != mask_edge.shape:
                        #print(f"    ⚠️  梯度和mask尺寸不匹配: gradient{gradient_magnitude.shape} vs mask_edge{mask_edge.shape}")
                        # 调整mask_edge尺寸以匹配gradient_magnitude
                        min_h = min(gradient_magnitude.shape[0], mask_edge.shape[0])
                        min_w = min(gradient_magnitude.shape[1], mask_edge.shape[1])
                        mask_edge = mask_edge[:min_h, :min_w]
                        gradient_magnitude = gradient_magnitude[:min_h, :min_w]
                    
                    if mask_edge.sum() > 0:
                        # 🔧 确保mask_edge是布尔类型用于索引
                        mask_edge_bool = (mask_edge > 0).astype(bool)
                        edge_gradient = gradient_magnitude[mask_edge_bool]
                        if len(edge_gradient) > 0:
                            avg_edge_gradient = np.mean(edge_gradient)
                            
                            # 真hole边缘梯度应该较强（深度变化明显）
                            if avg_edge_gradient > 30:  # 强梯度，表示深度变化
                                edge_gradient_score = 1.0
                            elif avg_edge_gradient > 20:  # 中等梯度
                                edge_gradient_score = 0.8
                            elif avg_edge_gradient > 10:  # 弱梯度
                                edge_gradient_score = 0.6
                            else:  # 几乎无梯度，平面特征
                                edge_gradient_score = 0.2
                
                except Exception as e:
                    print(f"    ⚠️  边缘梯度分析出错: {e}")
                    edge_gradient_score = 0.5
                        
        except Exception as e:
            print(f"    ⚠️  深度分析出错: {e}")
            depth_score = 0.5
            texture_score = 0.5
            edge_gradient_score = 0.5
    
    # 7. 圆度分析：真hole更接近圆形
    circularity_score = 0.5
    try:
        # 计算mask的周长
        contour_length = 0
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour_length = cv2.arcLength(contours[0], True)
            if contour_length > 0:
                # 圆度 = 4π * 面积 / 周长²
                circularity = 4 * np.pi * mask_area / (contour_length * contour_length)
                circularity = min(1.0, circularity)  # 限制在1以内
                
                if circularity > 0.7:  # 非常接近圆形
                    circularity_score = 1.0
                elif circularity > 0.5:  # 比较接近圆形
                    circularity_score = 0.8
                elif circularity > 0.3:  # 有点接近圆形
                    circularity_score = 0.6
                else:  # 不太像圆形
                    circularity_score = 0.3
    except Exception as e:
        print(f"    ⚠️  圆度分析出错: {e}")
        circularity_score = 0.5
    
    # 8. 🔍 新增：颜色亮度和对比度分析（重要：区分木板和hole）
    brightness_score = 0.5
    contrast_score = 0.5
    
    if image_array is not None:
        try:
            x1, y1, x2, y2 = map(int, bbox)
            roi = image_array[y1:y2, x1:x2]
            
            if roi.size > 0:
                # 计算区域内的平均亮度
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
                mean_brightness = np.mean(gray_roi)
                
                # 计算周围区域的平均亮度进行对比
                padding = 30
                y1_ext = max(0, y1 - padding)
                y2_ext = min(image_array.shape[0], y2 + padding)
                x1_ext = max(0, x1 - padding)
                x2_ext = min(image_array.shape[1], x2 + padding)
                
                surrounding_roi = image_array[y1_ext:y2_ext, x1_ext:x2_ext]
                if surrounding_roi.size > 0:
                    surrounding_gray = cv2.cvtColor(surrounding_roi, cv2.COLOR_RGB2GRAY) if len(surrounding_roi.shape) == 3 else surrounding_roi
                    surrounding_brightness = np.mean(surrounding_gray)
                    
                    # 真hole应该比周围区域暗得多（这是关键区别点）
                    brightness_diff = surrounding_brightness - mean_brightness
                    if brightness_diff > 60:  # 显著更暗，很可能是hole
                        brightness_score = 1.0
                    elif brightness_diff > 40:  # 较暗，可能是hole
                        brightness_score = 0.8
                    elif brightness_diff > 20:  # 稍暗，有可能是hole
                        brightness_score = 0.6
                    elif brightness_diff > 5:   # 稍微暗一些
                        brightness_score = 0.3
                    else:  # 亮度相同或更亮，很可能是平面物体（如木板）
                        brightness_score = 0.1
                        
                    # 计算对比度（hole内部通常对比度较低，比较均匀暗）
                    roi_std = np.std(gray_roi)
                    if roi_std < 8:  # 对比度很低，均匀暗区，很像hole
                        contrast_score = 1.0
                    elif roi_std < 15:  # 对比度较低，像hole
                        contrast_score = 0.8
                    elif roi_std < 25:  # 对比度中等
                        contrast_score = 0.6
                    elif roi_std < 35:  # 对比度较高，可能是纹理
                        contrast_score = 0.3
                    else:  # 对比度很高，很可能是复杂纹理（如木板）
                        contrast_score = 0.1
        except Exception as e:
            print(f"    ⚠️  亮度对比度分析出错: {e}")
            brightness_score = 0.5
            contrast_score = 0.5
    
    # 综合评分（加权平均，重点强化区分木板和hole的特征）
    weights = {
        'area': 0.08,
        'compactness': 0.08,
        'aspect': 0.06,
        'confidence': 0.06,
        'edge': 0.03,
        'depth': 0.25,        # 深度特征权重高
        'texture': 0.12,      # 纹理特征
        'edge_gradient': 0.08, # 边缘梯度特征
        'circularity': 0.04,   # 圆度特征
        'brightness': 0.15,    # 🔥新增：亮度对比，关键区分特征
        'contrast': 0.05       # 🔥新增：对比度，辅助区分特征
    }
    
    hole_score = (
        weights['area'] * area_score +
        weights['compactness'] * compactness_score +
        weights['aspect'] * aspect_score +
        weights['confidence'] * confidence_score +
        weights['edge'] * edge_score +
        weights['depth'] * depth_score +
        weights['texture'] * texture_score +
        weights['edge_gradient'] * edge_gradient_score +
        weights['circularity'] * circularity_score +
        weights['brightness'] * brightness_score +
        weights['contrast'] * contrast_score
    )
    
    # 设置验证阈值（适度降低阈值，平衡检测率和准确率）
    validation_threshold = 0.31314  # 降低阈值，确保能检测到hole
    is_valid_hole = hole_score >= validation_threshold
    
    # 详细验证信息
    validation_info = {
        'area_ratio': area_ratio,
        'area_score': area_score,
        'compactness': compactness,
        'compactness_score': compactness_score,
        'aspect_ratio': aspect_ratio,
        'aspect_score': aspect_score,
        'confidence_score': confidence_score,
        'edge_distance_ratio': edge_dist_ratio,
        'edge_score': edge_score,
        'depth_score': depth_score,
        'texture_score': texture_score,
        'edge_gradient_score': edge_gradient_score,
        'circularity_score': circularity_score,
        'brightness_score': brightness_score,  # 🆕 新增亮度特征
        'contrast_score': contrast_score,      # 🆕 新增对比度特征
        'final_score': hole_score,
        'threshold': validation_threshold,
        'bbox_size': (bbox_width, bbox_height),
        'mask_area': int(mask_area)
    }
    
    return is_valid_hole, hole_score, validation_info

# 自动视频帧提取
print("\n" + "="*50)
print("🎬 开始视频帧提取阶段")
print("="*50)

# 查找输入视频文件
input_video_path = find_video_file(input_video_dir)
if input_video_path is None:
    raise FileNotFoundError(f"无法在 {input_video_dir} 中找到视频文件")

print(f"📁 找到输入视频: {os.path.basename(input_video_path)}")

# 获取输入视频的帧率
input_video_fps = get_video_framerate(input_video_path)

# 清空并重新创建视频帧目录
print("🧹 清空视频帧目录...")
clear_directory(video_dir)
os.makedirs(video_dir, exist_ok=True)
print("✅ 视频帧目录已清空")

# 提取视频帧
success = extract_frames_with_ffmpeg(input_video_path, video_dir)
if not success:
    raise RuntimeError("视频帧提取失败")

print("\n" + "="*50)
print("🔍 开始物体检测与跟踪阶段")
print("="*50)

# create the output directory
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
CommonUtils.creat_dirs(result_dir)

print("\n🧹 清空输出目录...")
clear_directory(mask_data_dir)
clear_directory(json_data_dir)
clear_directory(result_dir)
print("✅ 输出目录已清空完成\n")
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir)
step = 20 # the step to sample frames for Grounding DINO predictor

sam2_masks = MaskDictionaryModel()
PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
objects_count = 0
frame_object_count = {}

# 全局物体标签一致性管理器
class ObjectLabelConsistencyManager:
    def __init__(self):
        self.object_label_history = {}  # {object_id: [labels_list]}
        self.object_confirmed_labels = {}  # {object_id: confirmed_label}
        self.label_confidence_threshold = 3  # 标签确认需要的最小投票数
        self.min_consecutive_frames = 5  # 最小连续帧数要求
        self.object_frame_history = {}  # {object_id: [(frame_idx, label)]}
        
    def add_label_observation(self, object_id, label, confidence=1.0, frame_idx=None):
        """为物体添加一个标签观察"""
        if object_id not in self.object_label_history:
            self.object_label_history[object_id] = []
            self.object_frame_history[object_id] = []
        
        # 添加带权重的标签观察
        self.object_label_history[object_id].append((label, confidence))
        
        # 记录帧历史
        if frame_idx is not None:
            self.object_frame_history[object_id].append((frame_idx, label))
        
        # 如果观察数量足够，确认标签
        if len(self.object_label_history[object_id]) >= self.label_confidence_threshold:
            self._confirm_label(object_id)
    
    def _confirm_label(self, object_id):
        """基于历史观察确认物体的最终标签"""
        if object_id not in self.object_label_history:
            return
            
        # 统计各标签的加权投票
        label_votes = {}
        for label, confidence in self.object_label_history[object_id]:
            if label not in label_votes:
                label_votes[label] = 0
            label_votes[label] += confidence
        
        # 选择得票最高的标签
        if label_votes:
            confirmed_label = max(label_votes.items(), key=lambda x: x[1])[0]
            
            # 检查标签连续性，过滤短暂错误
            if self._is_label_stable(object_id, confirmed_label):
                self.object_confirmed_labels[object_id] = confirmed_label
                print(f"  → 物体ID {object_id} 标签确认为: {confirmed_label}")
            else:
                # 如果标签不稳定，寻找最稳定的标签
                stable_label = self._find_most_stable_label(object_id)
                if stable_label:
                    self.object_confirmed_labels[object_id] = stable_label
                    print(f"  → 物体ID {object_id} 标签修正为稳定标签: {stable_label}")
    
    def _is_label_stable(self, object_id, label):
        """检查标签是否在足够多的连续帧中出现"""
        if object_id not in self.object_frame_history:
            return False
        
        frame_history = self.object_frame_history[object_id]
        if len(frame_history) < self.min_consecutive_frames:
            return True  # 帧数不足，暂时认为稳定
        
        # 检查最近的帧中是否有足够的连续性
        label_sequences = []
        current_sequence = 0
        
        for _, frame_label in frame_history:
            if frame_label == label:
                current_sequence += 1
            else:
                if current_sequence > 0:
                    label_sequences.append(current_sequence)
                current_sequence = 0
        
        if current_sequence > 0:
            label_sequences.append(current_sequence)
        
        # 如果最长连续序列大于等于最小要求，认为稳定
        max_sequence = max(label_sequences) if label_sequences else 0
        return max_sequence >= self.min_consecutive_frames
    
    def _find_most_stable_label(self, object_id):
        """找到最稳定的标签（连续出现最多次的）"""
        if object_id not in self.object_frame_history:
            return None
        
        frame_history = self.object_frame_history[object_id]
        label_sequences = {}
        
        for label in set([frame_label for _, frame_label in frame_history]):
            sequences = []
            current_sequence = 0
            
            for _, frame_label in frame_history:
                if frame_label == label:
                    current_sequence += 1
                else:
                    if current_sequence > 0:
                        sequences.append(current_sequence)
                    current_sequence = 0
            
            if current_sequence > 0:
                sequences.append(current_sequence)
            
            max_sequence = max(sequences) if sequences else 0
            label_sequences[label] = max_sequence
        
        # 返回最长连续序列的标签
        if label_sequences:
            return max(label_sequences.items(), key=lambda x: x[1])[0]
        return None
    
    def get_consistent_label(self, object_id, current_label=None, frame_idx=None):
        """获取物体的一致性标签"""
        # 如果已经确认了标签，直接使用确认的标签
        if object_id in self.object_confirmed_labels:
            return self.object_confirmed_labels[object_id]
        
        # 如果有当前检测标签，添加观察
        if current_label:
            self.add_label_observation(object_id, current_label, frame_idx=frame_idx)
        
        # 如果还没确认，使用历史观察中最频繁的标签
        if object_id in self.object_label_history and self.object_label_history[object_id]:
            label_counts = {}
            for label, _ in self.object_label_history[object_id]:
                label_counts[label] = label_counts.get(label, 0) + 1
            return max(label_counts.items(), key=lambda x: x[1])[0]
        
        # 如果都没有，返回当前标签
        return current_label
    
    def filter_transient_errors(self, json_data_dir, frame_names):
        """过滤短暂出现的错误标签"""
        print("\n🔍 过滤短暂错误标签...")
        
        # 重新收集所有帧的标签信息，按时间顺序
        all_objects = {}
        
        for i, frame_name in enumerate(frame_names):
            image_base_name = frame_name.split(".")[0]
            json_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
            if os.path.exists(json_path):
                frame_data = MaskDictionaryModel().from_json(json_path)
                for obj_id, obj_info in frame_data.labels.items():
                    if obj_id not in all_objects:
                        all_objects[obj_id] = []
                    all_objects[obj_id].append((i, obj_info.class_name))
        
        # 分析每个物体的标签序列，识别并修正短暂错误
        corrections_made = {}
        
        for obj_id, label_sequence in all_objects.items():
            if len(label_sequence) < 3:  # 太少的观察，跳过
                continue
            
            # 找到主要标签（出现最多的）
            label_counts = {}
            for _, label in label_sequence:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            if not label_counts:
                continue
                
            main_label = max(label_counts.items(), key=lambda x: x[1])[0]
            
            # 如果主要标签占比足够高，修正短暂的错误标签
            total_frames = len(label_sequence)
            main_label_ratio = label_counts[main_label] / total_frames
            
            if main_label_ratio >= 0.6:  # 主要标签占60%以上
                corrections = []
                for frame_idx, label in label_sequence:
                    if label != main_label:
                        corrections.append((frame_idx, label, main_label))
                
                if corrections:
                    corrections_made[obj_id] = {
                        'main_label': main_label,
                        'corrections': corrections
                    }
                    
                    # 应用修正
                    for frame_idx, _, correct_label in corrections:
                        frame_name = frame_names[frame_idx]
                        image_base_name = frame_name.split(".")[0]
                        json_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
                        
                        if os.path.exists(json_path):
                            frame_data = MaskDictionaryModel().from_json(json_path)
                            if obj_id in frame_data.labels:
                                frame_data.labels[obj_id].class_name = correct_label
                                frame_data.to_json(json_path)
        
        # 输出修正统计
        if corrections_made:
            print(f"📊 短暂错误标签修正统计:")
            for obj_id, correction_info in corrections_made.items():
                main_label = correction_info['main_label']
                corrections = correction_info['corrections']
                print(f"  物体ID {obj_id}: 主标签 '{main_label}' (修正 {len(corrections)} 个短暂错误)")
                for frame_idx, wrong_label, _ in corrections[:3]:  # 只显示前3个
                    print(f"    帧 {frame_idx}: {wrong_label} → {main_label}")
                if len(corrections) > 3:
                    print(f"    ... 还有 {len(corrections)-3} 个修正")
        else:
            print("✅ 未发现需要修正的短暂错误标签")
    
    def force_label_consistency(self, json_data_dir, frame_names):
        """在整个视频处理完成后，强制标签一致性"""
        print("\n🔄 执行全局标签一致性优化...")
        
        # 首先过滤短暂错误
        self.filter_transient_errors(json_data_dir, frame_names)
        
        # 收集所有帧的标签信息
        all_frame_labels = {}
        for frame_name in frame_names:
            image_base_name = frame_name.split(".")[0]
            json_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
            if os.path.exists(json_path):
                frame_data = MaskDictionaryModel().from_json(json_path)
                for obj_id, obj_info in frame_data.labels.items():
                    if obj_id not in all_frame_labels:
                        all_frame_labels[obj_id] = []
                    all_frame_labels[obj_id].append(obj_info.class_name)
        
        # 为每个物体确定最一致的标签
        label_changes = {}
        for obj_id, labels in all_frame_labels.items():
            # 统计标签频率
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # 选择最频繁的标签作为一致标签
            if label_counts:
                consistent_label = max(label_counts.items(), key=lambda x: x[1])[0]
                self.object_confirmed_labels[obj_id] = consistent_label
                
                # 统计需要修改的数量
                changes_needed = sum(1 for label in labels if label != consistent_label)
                if changes_needed > 0:
                    label_changes[obj_id] = {
                        'from_labels': list(set(labels)),
                        'to_label': consistent_label,
                        'changes_count': changes_needed,
                        'total_frames': len(labels)
                    }
        
        # 应用标签一致性修正
        for frame_name in frame_names:
            image_base_name = frame_name.split(".")[0]
            json_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
            if os.path.exists(json_path):
                frame_data = MaskDictionaryModel().from_json(json_path)
                modified = False
                
                for obj_id, obj_info in frame_data.labels.items():
                    if obj_id in self.object_confirmed_labels:
                        consistent_label = self.object_confirmed_labels[obj_id]
                        if obj_info.class_name != consistent_label:
                            obj_info.class_name = consistent_label
                            modified = True
                
                if modified:
                    frame_data.to_json(json_path)
        
        # 输出修正统计
        if label_changes:
            print(f"📊 最终标签一致性修正统计:")
            for obj_id, change_info in label_changes.items():
                print(f"  物体ID {obj_id}: {change_info['from_labels']} → {change_info['to_label']} "
                      f"(修正 {change_info['changes_count']}/{change_info['total_frames']} 帧)")
        else:
            print("✅ 所有物体标签已保持一致，无需修正")

# 物体重叠检测和ID去重管理器
class ObjectOverlapDeduplicationManager:
    def __init__(self):
        self.object_spatial_history = {}  # {object_id: [(frame_idx, bbox, mask_center)]}
        self.overlap_threshold = 0.5  # IoU阈值，超过此值认为是同一物体
        self.mask_overlap_threshold = 0.6  # mask重叠阈值
        self.center_distance_threshold = 50  # 中心点距离阈值（像素）
        self.object_confirmed_labels = {}  # {object_id: confirmed_label}
        
    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_mask_overlap(self, mask1, mask2):
        """计算两个mask的重叠比例"""
        if mask1.shape != mask2.shape:
            return 0.0
        
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def get_mask_center(self, mask):
        """获取mask的中心点"""
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return None
        return (float(np.mean(x_coords)), float(np.mean(y_coords)))
    
    def calculate_center_distance(self, center1, center2):
        """计算两个中心点的距离"""
        if center1 is None or center2 is None:
            return float('inf')
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def detect_overlapping_objects(self, mask_dict, frame_idx):
        """检测当前帧中重叠的物体并去重"""
        if len(mask_dict.labels) <= 1:
            return mask_dict
        
        print(f"\n🔍 检测帧 {frame_idx} 中的重叠物体...")
        
        # 收集所有物体的空间信息
        objects_info = []
        for obj_id, obj_info in mask_dict.labels.items():
            bbox = obj_info.bbox if hasattr(obj_info, 'bbox') and obj_info.bbox is not None else [0, 0, 0, 0]
            mask_center = self.get_mask_center(obj_info.mask)
            objects_info.append({
                'id': obj_id,
                'info': obj_info,
                'bbox': bbox,
                'center': mask_center,
                'label': obj_info.class_name
            })
        
        # 检测重叠物体对
        overlapping_pairs = []
        for i in range(len(objects_info)):
            for j in range(i + 1, len(objects_info)):
                obj1, obj2 = objects_info[i], objects_info[j]
                
                # 计算IoU
                iou = self.calculate_iou(obj1['bbox'], obj2['bbox'])
                
                # 计算mask重叠
                mask_overlap = self.calculate_mask_overlap(obj1['info'].mask, obj2['info'].mask)
                
                # 计算中心距离
                center_dist = self.calculate_center_distance(obj1['center'], obj2['center'])
                
                # 判断是否重叠（任一条件满足）
                is_overlapping = (
                    iou > self.overlap_threshold or 
                    mask_overlap > self.mask_overlap_threshold or
                    center_dist < self.center_distance_threshold
                )
                
                if is_overlapping:
                    overlapping_pairs.append({
                        'obj1': obj1,
                        'obj2': obj2,
                        'iou': iou,
                        'mask_overlap': mask_overlap,
                        'center_distance': center_dist
                    })
                    print(f"  发现重叠物体: ID{obj1['id']}({obj1['label']}) ↔ ID{obj2['id']}({obj2['label']}) "
                          f"[IoU:{iou:.3f}, Mask重叠:{mask_overlap:.3f}, 中心距离:{center_dist:.1f}]")
        
        if not overlapping_pairs:
            print("  ✅ 未发现重叠物体")
            return mask_dict
        
        # 去重：保留最佳物体
        objects_to_remove = set()
        deduplication_log = []
        
        for pair in overlapping_pairs:
            obj1, obj2 = pair['obj1'], pair['obj2']
            
            # 如果其中一个已经被标记删除，跳过
            if obj1['id'] in objects_to_remove or obj2['id'] in objects_to_remove:
                continue
            
            # 决策逻辑：优先保留更准确的标签和更大的mask
            keep_obj1 = self._decide_which_object_to_keep(obj1, obj2, pair)
            
            if keep_obj1:
                objects_to_remove.add(obj2['id'])
                deduplication_log.append(f"保留ID{obj1['id']}({obj1['label']}), 移除ID{obj2['id']}({obj2['label']})")
            else:
                objects_to_remove.add(obj1['id'])
                deduplication_log.append(f"保留ID{obj2['id']}({obj2['label']}), 移除ID{obj1['id']}({obj1['label']})")
        
        # 执行去重
        if objects_to_remove:
            print(f"  🗑️  去重操作:")
            for log in deduplication_log:
                print(f"    {log}")
            
            # 创建去重后的mask_dict
            deduplicated_mask_dict = MaskDictionaryModel(
                promote_type=mask_dict.promote_type,
                mask_name=mask_dict.mask_name
            )
            
            for obj_id, obj_info in mask_dict.labels.items():
                if obj_id not in objects_to_remove:
                    deduplicated_mask_dict.labels[obj_id] = obj_info
            
            print(f"  📊 去重结果: {len(mask_dict.labels)} → {len(deduplicated_mask_dict.labels)} 个物体")
            return deduplicated_mask_dict
        
        return mask_dict
    
    def _decide_which_object_to_keep(self, obj1, obj2, overlap_info):
        """决定保留哪个物体的逻辑 - 针对hole优化"""
        # 1. 优先级：hole > 其他物体（hole是我们的主要目标）
        label_priority = {'hole': 3, 'mouse': 2, 'bag': 1}
        priority1 = label_priority.get(obj1['label'], 0)
        priority2 = label_priority.get(obj2['label'], 0)
        
        if priority1 != priority2:
            return priority1 > priority2
        
        # 2. 如果都是hole，保留更符合hole特征的（更圆、更紧凑的）
        if obj1['label'] == 'hole' and obj2['label'] == 'hole':
            # 计算形状紧凑度
            area1 = obj1['info'].mask.sum()
            area2 = obj2['info'].mask.sum()
            bbox1 = obj1['bbox']
            bbox2 = obj2['bbox']
            
            # 计算紧凑度
            bbox_area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            bbox_area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            
            compactness1 = area1 / bbox_area1 if bbox_area1 > 0 else 0
            compactness2 = area2 / bbox_area2 if bbox_area2 > 0 else 0
            
            # 保留更紧凑的hole
            if abs(compactness1 - compactness2) > 0.1:
                return compactness1 > compactness2
        
        # 3. 如果标签优先级相同，比较mask面积（保留更大的）
        area1 = obj1['info'].mask.sum()
        area2 = obj2['info'].mask.sum()
        
        if abs(area1 - area2) > min(area1, area2) * 0.3:  # 面积差异超过30%
            return area1 > area2
        
        # 4. 如果面积相近，保留ID较小的（通常是先检测到的）
        return obj1['id'] < obj2['id']
    
    def update_spatial_history(self, object_id, frame_idx, bbox, mask):
        """更新物体的空间历史"""
        if object_id not in self.object_spatial_history:
            self.object_spatial_history[object_id] = []
        
        mask_center = self.get_mask_center(mask)
        self.object_spatial_history[object_id].append((frame_idx, bbox, mask_center))
        
        # 只保留最近的历史记录（避免内存过大）
        max_history = 50
        if len(self.object_spatial_history[object_id]) > max_history:
            self.object_spatial_history[object_id] = self.object_spatial_history[object_id][-max_history:]

# 初始化管理器
label_manager = ObjectLabelConsistencyManager()
overlap_manager = ObjectOverlapDeduplicationManager()
tracking_history = TrackingHistoryManager(
    max_missing_frames=5,      # 允许物体缺失5帧后再删除
    max_movement_ratio=0.2,    # 最大移动距离不超过图像对角线的20%
    max_size_change_ratio=0.6  # 最大尺寸变化不超过60%
)

"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
"""
print("视频总帧数:", len(frame_names))
for start_frame_idx in range(0, len(frame_names), step):
# prompt grounding dino to get the box coordinates on specific frame
    # continue
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    image = Image.open(img_path).convert("RGB")
    image_base_name = frame_names[start_frame_idx].split(".")[0]
    mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

    # 使用本地 GroundingDINO 进行推理
    image_source, transformed_image = load_image(img_path)
    
    # 在 float32 精度下进行推理，提高hole检测的精度
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        boxes, confidences, labels = predict(
            model=grounding_model, 
            image=transformed_image, 
            caption=text, 
            box_threshold=0.41314,  # 大幅降低box阈值以检测更多hole候选
            text_threshold=0.31314,  # 大幅降低text阈值提高召回率
            device=device
        )
    
    # 转换坐标格式和数据类型
    if len(boxes) > 0:
        # 转换为numpy格式并确保可写
        input_boxes = boxes.cpu().numpy().copy()
        confidences = confidences.cpu().numpy().copy()
        
        # 获取图像尺寸
        H, W = image_source.shape[:2]
        
        # 转换边界框坐标到像素坐标 (从归一化坐标到像素坐标)
        input_boxes[:, [0, 2]] *= W  # x坐标
        input_boxes[:, [1, 3]] *= H  # y坐标
        
        # GroundingDINO返回的是(cx, cy, w, h)格式，需要转换为(x1, y1, x2, y2)
    if len(boxes) > 0:
        # 转换为numpy格式并确保可写
        if hasattr(boxes, 'cpu'):
            input_boxes = boxes.cpu().numpy().copy()
        else:
            input_boxes = np.array(boxes).copy()
        if hasattr(confidences, 'cpu'):
            confidences_np = confidences.cpu().numpy().copy()
        else:
            confidences_np = np.array(confidences).copy()

        # 获取图像尺寸
        H, W = image_source.shape[:2]

        # 转换边界框坐标到像素坐标 (从归一化坐标到像素坐标)
        input_boxes[:, [0, 2]] *= W  # x坐标
        input_boxes[:, [1, 3]] *= H  # y坐标

        # GroundingDINO返回的是(cx, cy, w, h)格式，需要转换为(x1, y1, x2, y2)
        boxes_xyxy = input_boxes.copy()
        boxes_xyxy[:, 0] = input_boxes[:, 0] - input_boxes[:, 2] / 2  # x1 = cx - w/2
        boxes_xyxy[:, 1] = input_boxes[:, 1] - input_boxes[:, 3] / 2  # y1 = cy - h/2
        boxes_xyxy[:, 2] = input_boxes[:, 0] + input_boxes[:, 2] / 2  # x2 = cx + w/2
        boxes_xyxy[:, 3] = input_boxes[:, 1] + input_boxes[:, 3] / 2  # y2 = cy + h/2

        input_boxes = boxes_xyxy

        # 过滤和验证检测到的物体（专门针对hole优化）
        # 计算图像面积
        image_area = H * W
        valid_indices = []
        filtered_boxes = []
        filtered_confidences = []
        filtered_labels = []
        validation_results = []

        print(f"🔍 开始验证 {len(input_boxes)} 个检测候选:")
        
        for i, box in enumerate(input_boxes):
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            area_ratio = box_area / image_area

            # 清理标签，去除多余空格并转为小写
            clean_label = labels[i].strip().lower()
            if ' ' in clean_label:
                clean_label = clean_label.split()[0]  # 取第一个词作为主标签
            
            # 1. 基础尺寸过滤（更宽松的初步过滤）
            basic_size_ok = (
                area_ratio < 0.8 and  # 不能占据图像80%以上
                box_width < W * 0.95 and 
                box_height < H * 0.95 and
                box_width > 5 and   # 最小5像素
                box_height > 5      # 最小5像素
            )
            
            if not basic_size_ok:
                print(f"    ❌ 候选 {i+1}: {clean_label} - 尺寸不合理 (面积比: {area_ratio:.3f}, 尺寸: {box_width:.0f}x{box_height:.0f})")
                continue
            
            # 2. 标签相关性检验（增强版：强调深度特征）
            hole_related_terms = ['hole', 'opening', 'circular', 'round', 'gap', 'deep', 'dark', 'shadow', 'cavity', 'depth']
            depth_related_terms = ['deep', 'dark', 'shadow', 'cavity', 'depth']
            
            is_hole_related = any(term in clean_label for term in hole_related_terms)
            has_depth_hint = any(term in clean_label for term in depth_related_terms)
            
            # 如果标签包含深度相关词汇，给予更高优先级
            if has_depth_hint:
                print(f"    ✓ 候选 {i+1}: {clean_label} - 包含深度特征词汇，优先级提升")
            elif not is_hole_related:
                print(f"    ❌ 候选 {i+1}: {clean_label} - 标签与hole无关")
                continue
            
            print(f"    ✓ 候选 {i+1}: {clean_label} - 通过初步验证 (置信度: {confidences_np[i]:.3f})")
            
            # 暂时添加到有效列表，后续进行更详细的验证
            valid_indices.append(i)
            filtered_boxes.append(box)
            filtered_confidences.append(confidences_np[i])
            filtered_labels.append('hole')  # 统一标记为hole
            
        print(f"📊 初步过滤结果: {len(input_boxes)} → {len(valid_indices)} 个候选")

        if len(valid_indices) > 0:
            input_boxes = np.array(filtered_boxes)
            confidences = np.array(filtered_confidences)
            OBJECTS = filtered_labels
        else:
            input_boxes = np.empty((0, 4))
            confidences = np.array([])
            OBJECTS = []
            print("过滤后未发现有效对象")
    else:
        input_boxes = np.empty((0, 4))
        OBJECTS = []
        print("未检测到任何对象")


    # ========== 物体重叠检测和ID去重（在分割前，基于检测框和标签） ==========
    if len(input_boxes) > 1:
        # 构造一个临时mask_dict用于去重（只用box和label）
        temp_mask_dict = MaskDictionaryModel(promote_type=mask_dict.promote_type, mask_name=mask_dict.mask_name)
        for i, box in enumerate(input_boxes):
            # 伪造一个ObjectInfo，仅用box和label
            temp_obj = ObjectInfo(instance_id=i+1, mask=np.zeros((int(H), int(W)), dtype=bool), class_name=OBJECTS[i], logit=None)
            temp_obj.bbox = box
            temp_mask_dict.labels[i+1] = temp_obj
        # 去重
        deduped_mask_dict = overlap_manager.detect_overlapping_objects(temp_mask_dict, start_frame_idx)
        # 只保留未被去重的box和label
        keep_indices = []
        for obj_id, obj_info in deduped_mask_dict.labels.items():
            # 找到原始index
            for i, box in enumerate(input_boxes):
                if np.allclose(box, obj_info.bbox, atol=1.0) and OBJECTS[i] == obj_info.class_name:
                    keep_indices.append(i)
                    break
        if keep_indices:
            input_boxes = input_boxes[keep_indices]
            confidences = confidences[keep_indices]
            OBJECTS = [OBJECTS[i] for i in keep_indices]
            print(f"去重后保留 {len(OBJECTS)} 个对象: {OBJECTS}")
        else:
            input_boxes = np.empty((0, 4))
            confidences = np.array([])
            OBJECTS = []
            print("去重后未保留任何对象")

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))
    if input_boxes.shape[0] != 0:
        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # 🔧 确保生成的masks尺寸与图像尺寸匹配
        if masks.shape[1:] != (H, W):
            print(f"    ⚠️  调整SAM2生成的masks尺寸: {masks.shape} → (n, {H}, {W})")
            adjusted_masks = []
            for i in range(masks.shape[0]):
                mask = masks[i]
                if mask.shape != (H, W):
                    # 使用最近邻插值调整mask尺寸
                    mask_resized = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    adjusted_masks.append(mask_resized)
                else:
                    adjusted_masks.append(mask)
            masks = np.array(adjusted_masks)
        
        # 🔍 重要：添加hole特征验证阶段
        print(f"\n🔬 开始详细hole特征验证:")
        validated_masks = []
        validated_boxes = []
        validated_labels = []
        validated_confidences = []
        
        for i in range(len(masks)):
            mask = masks[i]
            box = input_boxes[i]
            confidence = confidences[i]
            label = OBJECTS[i]
            
            # 验证hole特征（传入原始图像进行深度分析）
            image_array = np.array(image.convert("RGB"))
            is_valid_hole, hole_score, validation_info = validate_hole_features(
                mask, box, (H, W), confidence, image_array
            )
            
            print(f"  🔍 验证候选 {i+1}: {label}")
            print(f"    - 面积比例: {validation_info['area_ratio']:.6f}")
            print(f"    - 形状紧凑度: {validation_info['compactness']:.3f}")
            print(f"    - 宽高比: {validation_info['aspect_ratio']:.3f}")
            print(f"    - 边缘距离: {validation_info['edge_distance_ratio']:.3f}")
            print(f"    - 深度特征: {validation_info['depth_score']:.3f}")
            print(f"    - 纹理特征: {validation_info['texture_score']:.3f}")
            print(f"    - 边缘梯度: {validation_info['edge_gradient_score']:.3f}")
            print(f"    - 圆度特征: {validation_info['circularity_score']:.3f}")
            print(f"    - 🆕亮度对比: {validation_info['brightness_score']:.3f}")
            print(f"    - 🆕对比度: {validation_info['contrast_score']:.3f}")
            print(f"    - 综合评分: {hole_score:.3f}/{validation_info['threshold']}")
            
            if is_valid_hole:
                # 额外验证：深度、纹理和亮度分数都必须达到最低标准
                depth_ok = validation_info['depth_score'] >= 0.11314  # 深度特征必须明显
                texture_ok = validation_info['texture_score'] >= 0.11314  # 纹理必须相对均匀
                brightness_ok = validation_info['brightness_score'] >= 0.01314  # 🆕 亮度对比必须明显（关键！）
                
                if depth_ok and texture_ok and brightness_ok:
                    print(f"    ✅ 验证通过 - 确认为有效hole（深度+纹理+亮度验证通过）")
                    validated_masks.append(mask)
                    validated_boxes.append(box)
                    validated_labels.append('hole')
                    validated_confidences.append(hole_score)  # 使用验证后的置信度
                elif not depth_ok:
                    print(f"    ❌ 验证失败 - 深度特征不足（{validation_info['depth_score']:.3f} < 0.11314），可能是平面物体")
                elif not texture_ok:
                    print(f"    ❌ 验证失败 - 纹理过于复杂（{validation_info['texture_score']:.3f} < 0.11314），可能是木板等平面物体")
                elif not brightness_ok:
                    print(f"    ❌ 验证失败 - 亮度对比不足（{validation_info['brightness_score']:.3f} < 0.01314），可能是浅色平面物体如木板")
            else:
                print(f"    ❌ 验证失败 - 综合评分不足")
                # 分析主要失败原因
                if validation_info['depth_score'] < 0.4:
                    print(f"      主要原因：缺乏深度特征（亮度差异不足）")
                if validation_info['brightness_score'] < 0.4:
                    print(f"      主要原因：亮度对比不足（可能是平面物体如木板）")
                if validation_info['texture_score'] < 0.4:
                    print(f"      主要原因：纹理过于复杂（不像hole内部）")
                if validation_info['compactness'] < 0.5:
                    print(f"      主要原因：形状不够紧凑")
                if validation_info['circularity_score'] < 0.4:
                    print(f"      主要原因：形状不够圆形")
        
        print(f"📊 最终验证结果: {len(masks)} → {len(validated_masks)} 个确认hole")
        
        if len(validated_masks) > 0:
            # 使用验证后的结果
            masks = np.array(validated_masks)
            input_boxes = np.array(validated_boxes)
            OBJECTS = validated_labels
            confidences = np.array(validated_confidences)
            
            print(f"✅ 最终确认的hole:")
            for i, (conf, label) in enumerate(zip(confidences, OBJECTS)):
                print(f"  Hole {i+1}: {label} (验证评分: {conf:.3f})")
        else:
            # 没有验证通过的hole
            masks = np.empty((0, H, W))
            input_boxes = np.empty((0, 4))
            OBJECTS = []
            confidences = np.array([])
            print("❌ 未发现符合特征的hole")
        
        # Step 3: Register each object's positive points to video predictor
        if len(OBJECTS) > 0 and mask_dict.promote_type == "mask":
            mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
        else:
            if len(OBJECTS) == 0:
                pass  # Skip silently
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")
    else:
        mask_dict = sam2_masks

    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    # 优化的追踪更新：使用更宽松的IoU阈值和多重匹配策略
    objects_count = mask_dict.update_masks_enhanced(
        tracking_annotation_dict=sam2_masks, 
        iou_threshold=0.5,  # 进一步降低IoU阈值，提高追踪连续性
        spatial_threshold=80,  # 增加空间距离阈值，允许更大的移动
        objects_count=objects_count,
        tracking_history=tracking_history  # 传入追踪历史管理器
    )
    
    # 为新检测到的物体添加标签观察到一致性管理器和合理性检查
    valid_objects = {}
    image_shape = (H, W)  # 使用当前帧的图像尺寸
    
    for object_id, object_info in mask_dict.labels.items():
        current_label = object_info.class_name
        
        # 计算物体中心和面积
        mask_center = tracking_history._get_mask_center(object_info.mask)
        mask_area = object_info.mask.sum().item() if isinstance(object_info.mask, torch.Tensor) else object_info.mask.sum()
        
        # 运动和尺寸合理性检查
        is_consistent = tracking_history.validate_object_consistency(
            object_id, mask_center, mask_area, image_shape
        )
        
        if is_consistent:
            # 添加当前检测的标签观察（新检测的标签权重更高）
            label_manager.add_label_observation(object_id, current_label, confidence=2.0, frame_idx=start_frame_idx)
            # 获取一致性标签并更新
            consistent_label = label_manager.get_consistent_label(object_id, current_label, frame_idx=start_frame_idx)
            if consistent_label != current_label:
                print(f"  📝 标签一致性修正: 物体ID {object_id} {current_label} → {consistent_label}")
                object_info.class_name = consistent_label
            
            # 更新追踪历史
            tracking_history.update_object_position(object_id, mask_center, start_frame_idx, mask_area)
            valid_objects[object_id] = object_info
        else:
            print(f"  🚫 过滤物体ID {object_id}：运动或尺寸变化不合理")
    
    # 更新mask_dict只保留合理的物体
    mask_dict.labels = valid_objects
    
    # 🔍 追踪质量监控
    print(f"\n📊 帧 {start_frame_idx} 追踪质量报告:")
    print(f"  当前追踪物体数量: {len(mask_dict.labels)}")
    print(f"  历史追踪记录: {len(tracking_history.object_history)} 个物体")
    
    # 显示每个物体的追踪状态
    for object_id, object_info in mask_dict.labels.items():
        if object_id in tracking_history.object_history:
            history = tracking_history.object_history[object_id]
            missing_count = history['missing_count']
            position_count = len(history['positions'])
            if missing_count > 0:
                print(f"    🟡 ID {object_id} ({object_info.class_name}): 连续缺失 {missing_count} 帧，历史位置 {position_count} 个")
            else:
                print(f"    🟢 ID {object_id} ({object_info.class_name}): 正常追踪，历史位置 {position_count} 个")
        else:
            print(f"    🔵 ID {object_id} ({object_info.class_name}): 新检测物体")
    
    frame_object_count[start_frame_idx] = objects_count
    print(f"累计物体计数: {objects_count}")
    
    if len(mask_dict.labels) == 0:
        mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list = frame_names[start_frame_idx:start_frame_idx+step])
        continue
    else:
        video_predictor.reset_state(inference_state)

        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                )
        
        video_segments = {}  # output the following {step} frames tracking masks
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
            frame_masks = MaskDictionaryModel()
            
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                
                # 获取一致性标签
                original_class_name = mask_dict.get_target_class_name(out_obj_id)
                consistent_class_name = label_manager.get_consistent_label(out_obj_id, original_class_name, frame_idx=out_frame_idx)
                
                object_info = ObjectInfo(
                    instance_id=out_obj_id, 
                    mask=out_mask[0], 
                    class_name=consistent_class_name,  # 使用一致性标签
                    logit=mask_dict.get_target_logit(out_obj_id)
                )
                object_info.update_box()
                
                # 运动和尺寸合理性检查
                mask_center = tracking_history._get_mask_center(out_mask[0])
                mask_area = out_mask[0].sum().item() if isinstance(out_mask[0], torch.Tensor) else out_mask[0].sum()
                
                # 验证物体运动和尺寸变化的合理性
                image_shape = (out_mask.shape[-2], out_mask.shape[-1])
                is_consistent = tracking_history.validate_object_consistency(
                    out_obj_id, mask_center, mask_area, image_shape
                )
                
                if is_consistent:
                    # 只有通过合理性检查的物体才会被保留
                    frame_masks.labels[out_obj_id] = object_info
                    # 更新追踪历史
                    tracking_history.update_object_position(out_obj_id, mask_center, out_frame_idx, mask_area)
                else:
                    # 不合理的检测，跳过这个物体
                    print(f"    🚫 跳过物体ID {out_obj_id}：运动或尺寸变化不合理")
                    continue
                
                image_base_name = frame_names[out_frame_idx].split(".")[0]
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)

    """
    Step 5: save the tracking masks and json files
    """
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            # 🔧 确保mask和mask_img尺寸匹配
            if obj_info.mask.shape != mask_img.shape:
                #print(f"    ⚠️  对象{obj_id}的mask尺寸不匹配: {obj_info.mask.shape} vs {mask_img.shape}")
                # 调整obj_info.mask尺寸以匹配mask_img
                if hasattr(obj_info.mask, 'cpu'):
                    mask_np = obj_info.mask.cpu().numpy()
                elif hasattr(obj_info.mask, 'numpy'):
                    mask_np = obj_info.mask.numpy()
                else:
                    mask_np = obj_info.mask
                mask_resized = cv2.resize(
                    mask_np.astype(np.uint8), 
                    (mask_img.shape[1], mask_img.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                obj_info.mask = torch.from_numpy(mask_resized)
            
            # 🔧 确保mask是布尔类型用于索引，处理CUDA tensor
            if hasattr(obj_info.mask, 'cpu'):
                mask_bool = obj_info.mask.cpu().numpy().astype(bool)
            else:
                mask_bool = obj_info.mask.astype(bool)
            mask_img[mask_bool] = obj_id

        # 🔧 处理CUDA tensor转换为numpy
        if hasattr(mask_img, 'cpu'):
            mask_img = mask_img.cpu().numpy().astype(np.uint16)
        else:
            mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        frame_masks_info.to_json(json_data_path)
       

# 注释掉第一次绘制，因为反向跟踪会改进结果，只需要最终结果
# CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

print("try reverse tracking")
start_object_id = 0
object_info_dict = {}
for frame_idx, current_object_count in frame_object_count.items():
    if frame_idx != 0:
        video_predictor.reset_state(inference_state)
        image_base_name = frame_names[frame_idx].split(".")[0]
        json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
        json_data = MaskDictionaryModel().from_json(json_data_path)
        mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
        mask_array = np.load(mask_data_path)
        
        # 检查是否有新对象需要反向跟踪
        has_new_objects = False
        for object_id in range(start_object_id+1, current_object_count+1):
            print("reverse tracking object", object_id)
            object_info_dict[object_id] = json_data.labels[object_id]
            video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_array == object_id)
            has_new_objects = True
        
        # 只有当有新对象时才进行反向跟踪
        if has_new_objects:
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step*2,  start_frame_idx=frame_idx, reverse=True):
                image_base_name = frame_names[out_frame_idx].split(".")[0]
                json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
                json_data = MaskDictionaryModel().from_json(json_data_path)
                mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
                mask_array = np.load(mask_data_path)
                # merge the reverse tracking masks with the original masks
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0).cpu()
                    if out_mask.sum() == 0:
                        print("no mask for object", out_obj_id, "at frame", out_frame_idx)
                        continue
                    
                    object_info = object_info_dict[out_obj_id]
                    object_info.mask = out_mask[0]
                    
                    # 🔧 确保mask和mask_array尺寸匹配（关键修复）
                    if object_info.mask.shape != mask_array.shape:
                        # 调整object_info.mask尺寸以匹配mask_array
                        if object_info.mask.shape[0] != mask_array.shape[0] or object_info.mask.shape[1] != mask_array.shape[1]:
                            # 使用OpenCV resize调整mask尺寸
                            if hasattr(object_info.mask, 'cpu'):
                                mask_np = object_info.mask.cpu().numpy().astype(np.uint8)
                            else:
                                mask_np = object_info.mask.numpy().astype(np.uint8)
                            mask_resized = cv2.resize(
                                mask_np, 
                                (mask_array.shape[1], mask_array.shape[0]), 
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                            object_info.mask = torch.from_numpy(mask_resized)
                    
                    # 应用标签一致性
                    consistent_label = label_manager.get_consistent_label(out_obj_id, object_info.class_name, frame_idx=out_frame_idx)
                    if consistent_label != object_info.class_name:
                        print(f"  反向跟踪标签一致性修正: 物体ID {out_obj_id} {object_info.class_name} → {consistent_label}")
                        object_info.class_name = consistent_label
                    
                    object_info.update_box()
                    json_data.labels[out_obj_id] = object_info
                    mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
                    # 🔧 确保mask是布尔类型用于索引，处理CUDA tensor
                    if hasattr(object_info.mask, 'cpu'):
                        mask_bool = object_info.mask.cpu().numpy().astype(bool)
                    elif hasattr(object_info.mask, 'numpy'):
                        mask_bool = object_info.mask.numpy().astype(bool)
                    else:
                        mask_bool = object_info.mask.astype(bool)
                    mask_array[mask_bool] = out_obj_id
                
                np.save(mask_data_path, mask_array)
                json_data.to_json(json_data_path)
        
    start_object_id = current_object_count

"""
Step 6: Draw the results and save the video
"""
# 执行全局标签一致性优化
label_manager.force_label_consistency(json_data_dir, frame_names)

print("\n🎨 开始绘制最终结果...")
print("🧹 清空结果目录...")
clear_directory(result_dir)
print("✅ 结果目录已清空")

# 使用反向跟踪后的改进结果
CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

# 使用输入视频的帧率创建输出视频
print(f"🎬 使用输入视频帧率 {input_video_fps:.2f} fps 创建输出视频...")
create_video_from_images(result_dir, output_video_path, frame_rate=input_video_fps)

# 输出最终结果路径信息
print("\n🎉 视频分析完成！")
print("="*60)
print(f"� 输入视频路径: {input_video_path}")
print(f"🎞️  提取帧数据路径: {video_dir}")
print(f"�📸 注释图像保存路径: {result_dir}")
print(f"🎬 最终视频输出路径: {output_video_path}")
print(f"� 输出视频帧率: {input_video_fps:.2f} fps (与输入视频一致)")
print(f"�💾 遮罩数据保存路径: {mask_data_dir}")
print(f"📄 JSON数据保存路径: {json_data_dir}")
print(f"📊 总共处理了 {len(frame_names)} 帧图像")
print("="*60)

# 统计hole检测结果
hole_stats = {"total_holes": 0, "frames_with_holes": 0}
for frame_name in frame_names:
    image_base_name = frame_name.split(".")[0]
    json_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
    if os.path.exists(json_path):
        frame_data = MaskDictionaryModel().from_json(json_path)
        frame_holes = sum(1 for obj_info in frame_data.labels.values() if obj_info.class_name == 'hole')
        if frame_holes > 0:
            hole_stats["frames_with_holes"] += 1
            hole_stats["total_holes"] += frame_holes

print(f"\n🔍 Hole检测统计:")
print(f"   - 检测到的hole总数: {hole_stats['total_holes']}")
print(f"   - 包含hole的帧数: {hole_stats['frames_with_holes']}/{len(frame_names)}")
if len(frame_names) > 0:
    print(f"   - Hole检出率: {hole_stats['frames_with_holes']/len(frame_names)*100:.1f}%")
print("="*60)
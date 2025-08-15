import os
import json
import cv2
import albumentations as A
from tqdm import tqdm
import warnings

def create_aug_pipeline():
    warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
    
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-30, 30),
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0)
        ], p=0.3),
        A.OneOf([
            A.GlassBlur(sigma=0.7, max_delta=4, p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0)
        ], p=0.3)
    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False
    ))

def load_annotation(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    shapes = data.get('shapes', [])
    all_keypoints = []
    shape_info = []  # 存储(shape_type, label, point_count)
    
    for shape in shapes:
        shape_type = shape['shape_type']
        label = shape['label']
        points = shape['points']
        
        shape_info.append((shape_type, label, len(points)))
        all_keypoints.extend(points)
    
    return data, all_keypoints, shape_info

def clamp_relative_coordinates(points, image_width, image_height):
    """
    将绝对坐标转换为相对坐标并限制在[0, 1]范围内
    步骤：
    1. 绝对坐标 -> 相对坐标（x/width, y/height）
    2. 限制相对坐标在0到1之间
    3. 相对坐标 -> 绝对坐标（用于保存标注）
    """
    clamped_points = []
    for (x, y) in points:
        # 转换为相对坐标（归一化）
        x_rel = x / image_width
        y_rel = y / image_height
        
        # 核心：限制在[0, 1]范围内（YOLO要求）
        x_rel_clamped = max(0.0, min(1.0, x_rel))
        y_rel_clamped = max(0.0, min(1.0, y_rel))
        
        # 转回绝对坐标（保持标注文件格式兼容）
        x_clamped = x_rel_clamped * image_width
        y_clamped = y_rel_clamped * image_height
        
        clamped_points.append([round(x_clamped, 6), round(y_clamped, 6)])  # 保留6位小数
    return clamped_points

def rebuild_shapes(transformed_keypoints, shape_info, image_width, image_height):
    """重组形状并确保坐标相对值在[0, 1]范围内"""
    shapes = []
    current_idx = 0
    
    for shape_type, label, point_count in shape_info:
        # 提取当前形状的所有点
        points = transformed_keypoints[current_idx:current_idx + point_count]
        current_idx += point_count
        
        # 限制坐标范围（核心修改）
        clamped_points = clamp_relative_coordinates(points, image_width, image_height)
        
        shapes.append({
            'label': label,
            'points': clamped_points,  # 使用限制后的坐标
            'shape_type': shape_type,
            'flags': {}
        })
    
    return shapes

def update_annotation(original_anno, transformed_kps, shape_info, img_name, image_width, image_height):
    """更新标注文件，包含坐标范围限制"""
    new_anno = original_anno.copy()
    new_anno['imagePath'] = img_name
    new_anno['imageWidth'] = image_width 
    new_anno['imageHeight'] = image_height
    new_anno['shapes'] = rebuild_shapes(transformed_kps, shape_info, image_width, image_height)
    return new_anno

def process_single_pair(img_path, json_path, out_img_dir, out_labels_dir, aug_pipeline, num_augs=4):
    """处理单组图片和标注文件，确保坐标符合YOLO要求"""
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告: 无法读取图片 {img_path}，已跳过")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = img.shape[:2]  # 原图宽高
    
    # 读取标注
    try:
        original_anno, keypoints, shape_info = load_annotation(json_path)
    except Exception as e:
        print(f"警告: 处理标注 {json_path} 出错: {str(e)}，已跳过")
        return
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    img_ext = os.path.splitext(img_path)[1]
    
    # 保存原始文件（同时检查原始坐标是否合规）
    orig_img_dest = os.path.join(out_img_dir, f"{base_name}{img_ext}")
    orig_labels_dest = os.path.join(out_labels_dir, f"{base_name}.json")
    
    if not os.path.exists(orig_img_dest):
        cv2.imwrite(orig_img_dest, img)
    if not os.path.exists(orig_labels_dest):
        # 原始标注也需要检查坐标范围
        original_anno_clamped = update_annotation(
            original_anno, keypoints, shape_info, 
            f"{base_name}{img_ext}", orig_width, orig_height
        )
        with open(orig_labels_dest, 'w', encoding='utf-8') as f:
            json.dump(original_anno_clamped, f, ensure_ascii=False, indent=2)
    
    # 生成增强版本
    for i in range(num_augs):
        try:
            augmented = aug_pipeline(image=img_rgb, keypoints=keypoints)
        except Exception as e:
            print(f"警告: 增强 {base_name} 时出错: {str(e)}，已跳过当前增强版本")
            continue
        
        # 增强后图像的宽高（通常与原图一致，除非有缩放但此处未剪裁）
        aug_height, aug_width = augmented['image'].shape[:2]
        
        # 增强文件命名
        aug_suffix = f"_aug_{i+1}"
        aug_img_name = f"{base_name}{aug_suffix}{img_ext}"
        aug_labels_name = f"{base_name}{aug_suffix}.json"
        
        # 保存增强图片
        aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_img_dir, aug_img_name), aug_img_bgr)
        
        # 保存增强标注（带坐标限制）
        updated_anno = update_annotation(
            original_anno,
            augmented['keypoints'],  # 增强后的关键点（绝对坐标）
            shape_info,
            aug_img_name,
            aug_width,  # 增强后图像宽度
            aug_height  # 增强后图像高度
        )
        with open(os.path.join(out_labels_dir, aug_labels_name), 'w', encoding='utf-8') as f:
            json.dump(updated_anno, f, ensure_ascii=False, indent=2)

def batch_process_det(det_dir, out_det_dir, num_augs=4):
    """批量处理det目录下的所有数据，确保坐标符合YOLO要求"""
    in_labels_dir = os.path.join(det_dir, "labels")
    in_images_dir = os.path.join(det_dir, "images")
    
    if not os.path.exists(in_labels_dir):
        raise ValueError(f"标注目录不存在: {in_labels_dir}")
    if not os.path.exists(in_images_dir):
        raise ValueError(f"图片目录不存在: {in_images_dir}")
    
    out_images_dir = os.path.join(out_det_dir, "images")
    out_labels_dir = os.path.join(out_det_dir, "labels")
    
    print(f"正在处理图片目录: {in_images_dir}")
    print(f"正在处理标注目录: {in_labels_dir}")
    
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    img_files = [f for f in os.listdir(in_images_dir) if f.lower().endswith(img_extensions)]
    
    print(f"找到 {len(img_files)} 个图片文件")
    
    processed_count = 0
    skipped_count = 0
    
    aug_pipeline = create_aug_pipeline()
    for img_file in tqdm(img_files, desc="处理进度"):
        img_path = os.path.join(in_images_dir, img_file)
        json_file = os.path.splitext(img_file)[0] + '.json'
        json_path = os.path.join(in_labels_dir, json_file)
        
        if not os.path.exists(json_path):
            print(f"警告: 图片 {img_file} 无对应标注文件 {json_file}，已跳过")
            skipped_count += 1
            continue
        
        process_single_pair(
            img_path, json_path,
            out_images_dir, out_labels_dir,
            aug_pipeline, num_augs
        )
        processed_count += 1
    
    return processed_count, skipped_count, len(img_files)

if __name__ == "__main__":
    DET_INPUT_DIR = "../../datasets/detection"               # 输入目录
    DET_OUTPUT_DIR = "../../datasets/detection1"    # 输出目录
    AUGMENT_TIMES = 4                            # 增强次数（+1原图=5倍）
    
    try:
        processed, skipped, total_found = batch_process_det(DET_INPUT_DIR, DET_OUTPUT_DIR, AUGMENT_TIMES)
        print(f"\n处理完成！结果保存至 {DET_OUTPUT_DIR}")
        print(f"总数据量为原始有效数据的 {AUGMENT_TIMES + 1} 倍")
        print(f"处理统计: 共找到 {total_found} 个图片文件，成功处理 {processed} 组数据，跳过 {skipped} 张无标注图片")
        print(f"所有标注坐标已确保相对值在0-1之间，兼容YOLO训练")
    except Exception as e:
        print(f"执行出错: {str(e)}")
    

    
    
    
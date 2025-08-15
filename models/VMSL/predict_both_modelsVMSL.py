import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


class Config:
    img_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "."
    csv_file = "../../../../datasets/data.csv"
    det_json_dir = "../../../../datasets/detection"
    seg_json_dir = "../../../../datasets/segmentation"
    
    # 移除VMTL模型路径
    yolo_model_path = "../../../../outputs/checkpoints/proposed/best_yolo_convnext_model.pth"
    clinic_yolo_path = "../../../../outputs/YOLO/results/detect/YOLOV12/weights/best.pt"
    wood_yolo_path = "../../../../outputs/YOLO/results/segment/YOLOV12/weights/best.pt"


    hidden_dim = 384
    num_classes = 2
    dropout_rate = 0.5
    detection_num_classes = 1
    
  
    output_csv = "predictions_results.csv"
    batch_size = 4

print(f"Using device: {Config.device}")


class YOLOROIExtractor:
    def __init__(self, clinic_model_path, wood_model_path, device):
        self.device = device
        self.clinic_model = YOLO(clinic_model_path)
        self.wood_model = YOLO(wood_model_path)

    def extract_clinic_roi(self, image, confidence_threshold=0.5):
        results = self.clinic_model(image, conf=confidence_threshold, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return image
        boxes = results[0].boxes
        best_box_idx = torch.argmax(boxes.conf)
        best_box = boxes.xyxy[best_box_idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = best_box
        roi = image[y1:y2, x1:x2]
        if roi.shape[0] < 32 or roi.shape[1] < 32:
            return image
        return roi

    def extract_wood_roi(self, image, confidence_threshold=0.5):
        results = self.wood_model(image, conf=confidence_threshold, verbose=False)
        if len(results) == 0 or results[0].masks is None:
            return image
        masks = results[0].masks.data.cpu().numpy()
        if len(masks) == 0:
            return image
        mask = masks[0]
        coords = np.where(mask > 0.5)
        if len(coords[0]) == 0:
            return image
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        roi = image[y_min:y_max, x_min:x_max]
        if roi.shape[0] < 32 or roi.shape[1] < 32:
            return image
        return roi

# ConvQKV Attention Module (YOLO模型仍在使用)
class ConvQKVAttention(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_heads=8, num_iterations=3, sfs_scale=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_heads = num_heads
        self.num_iterations = num_iterations
        self.sfs_scale = sfs_scale
        self.sfs_avg_pool = nn.AvgPool2d(kernel_size=sfs_scale, stride=sfs_scale)
        self.sfs_max_pool = nn.MaxPool2d(kernel_size=sfs_scale, stride=sfs_scale)
        self.sfs_lambda = nn.Parameter(torch.tensor(0.5))
        self.q_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.o_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        )
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(self.out_channels * 2, self.out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.delta = nn.Parameter(torch.tensor(1.0))

    def _sfs(self, x):
        avg_pool = self.sfs_avg_pool(x)
        max_pool = self.sfs_max_pool(x)
        mixed_pool = self.sfs_lambda * avg_pool + (1 - self.sfs_lambda) * max_pool
        return mixed_pool

    def _cfe(self, x_main, x_aux):
        x_main = self._sfs(x_main)
        x_aux = self._sfs(x_aux)
        B, C, H, W = x_main.shape
        head_dim = self.out_channels // self.num_heads
        q = self.q_conv(x_aux).view(B, self.num_heads, head_dim, H * W)
        k = self.k_conv(x_main).view(B, self.num_heads, head_dim, H * W)
        v = self.v_conv(x_main).view(B, self.num_heads, head_dim, H * W)
        attention_scores = torch.matmul(q.transpose(2, 3), k.transpose(2, 3).transpose(-2, -1)) / (head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        z = torch.matmul(attention_weights, v.transpose(2, 3))
        z = z.transpose(2, 3).contiguous().view(B, -1, H, W)
        z = self.o_conv(z)
        t_prime = self.alpha * z + self.beta * x_main
        t_enhanced = self.gamma * t_prime + self.delta * self.ffn(t_prime)
        t_enhanced = F.interpolate(t_enhanced, scale_factor=self.sfs_scale, mode='bilinear', align_corners=False)
        return t_enhanced, attention_weights, z

    def forward(self, x1, x2):
        t_clinic = x1
        t_wood = x2
        for _ in range(self.num_iterations - 1):
            t_clinic, _, _ = self._cfe(t_clinic, t_wood)
            t_wood, _, _ = self._cfe(t_wood, t_clinic)
        t_clinic, clinic_attention, z_c = self._cfe(t_clinic, t_wood)
        t_wood, wood_attention, z_w = self._cfe(t_wood, t_clinic)
        fused_features = self.fusion_layers(torch.cat([t_clinic, t_wood], dim=1))
        z_c = z_c.mean(dim=(2, 3)) if len(z_c.shape) == 4 else z_c
        z_w = z_w.mean(dim=(2, 3)) if len(z_w.shape) == 4 else z_w
        z_f = fused_features.mean(dim=(2, 3))
        return fused_features, (clinic_attention, wood_attention), (z_c, z_w, z_f)

# YOLO-ConvNeXt Model
class YOLOConvNeXtV2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = timm.create_model(
            'convnextv2_tiny',
            pretrained=True,
            features_only=True,
            out_indices=[0, 2, 3]
        )
        for param in self.encoder.parameters():
            param.requires_grad = False
        for stage in [self.encoder.stages_2, self.encoder.stages_3]:
            for param in stage.parameters():
                param.requires_grad = True
        encoder_channels = self.encoder.feature_info.channels()
        self.encoder_projs = nn.ModuleList([
            nn.Conv2d(ch, config.hidden_dim, kernel_size=1) for ch in encoder_channels
        ])
        self.cross_attention = ConvQKVAttention(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            num_heads=8,
            num_iterations=3
        )
        self.head_f = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )

    def forward(self, c_img, w_img):
        c_features = self.encoder(c_img)
        w_features = self.encoder(w_img)
        c_projected = [proj(feat) for proj, feat in zip(self.encoder_projs, c_features)]
        w_projected = [proj(feat) for proj, feat in zip(self.encoder_projs, w_features)]
        c_encoded = c_projected[-1]
        w_encoded = w_projected[-1]
        fused_features, (clinic_attention, wood_attention), (z_c, z_w, z_f) = self.cross_attention(c_encoded, w_encoded)
        cls_pred_f = self.head_f(z_f)
        return {
            'cls_pred_f': cls_pred_f
        }

# Dataset for prediction (简化，只保留YOLO需要的ROI数据)
class VitiligoDataset(Dataset):
    def __init__(self, root_dir, csv_file, det_json_dir, seg_json_dir, roi_extractor, 
                 transform=None, image_size=(224, 224)):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.det_json_dir = det_json_dir
        self.seg_json_dir = seg_json_dir
        self.roi_extractor = roi_extractor
        self.image_size = image_size
        self.transform = transform
        self.grouped = self.data.groupby(['pair_id', 'stability'])
        self.group_keys = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        pair_id, stability = self.group_keys[idx]
        pairs = self.grouped.get_group((pair_id, stability))
        stability_label = 0 if stability.lower() == 'stable' else 1
        clinic = pairs[pairs['image_type'] == 'clinic']
        wood = pairs[pairs['image_type'] == 'wood']
        
        if clinic.empty or wood.empty:
            zero_tensor = torch.zeros(3, *self.image_size, dtype=torch.float32)
            return zero_tensor, zero_tensor, stability_label, pair_id, stability
        
        clinic_path = os.path.join(self.root_dir, clinic['image_path'].values[0])
        wood_path = os.path.join(self.root_dir, wood['image_path'].values[0])
        
        try:
            clinic_image = Image.open(clinic_path).convert('RGB').resize(self.image_size)
            wood_image = Image.open(wood_path).convert('RGB').resize(self.image_size)
            clinic_np = np.array(clinic_image)
            wood_np = np.array(wood_image)
            
            # 提取ROI
            try:
                clinic_roi = self.roi_extractor.extract_clinic_roi(clinic_np)
                wood_roi = self.roi_extractor.extract_wood_roi(wood_np)
                clinic_roi = cv2.resize(clinic_roi, self.image_size)
                wood_roi = cv2.resize(wood_roi, self.image_size)
            except:
                clinic_roi = clinic_np
                wood_roi = wood_np
            
            # 应用变换
            if self.transform:
                yolo_aug = self.transform(image=clinic_roi, image1=wood_roi)
                clinic_roi_tensor = yolo_aug['image']
                wood_roi_tensor = yolo_aug['image1']
            else:
                clinic_roi_tensor = torch.tensor(clinic_roi.transpose(2, 0, 1), dtype=torch.float32) / 255.0
                wood_roi_tensor = torch.tensor(wood_roi.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            
            return clinic_roi_tensor, wood_roi_tensor, stability_label, pair_id, stability
            
        except Exception as e:
            print(f"Error loading {pair_id}: {e}")
            zero_tensor = torch.zeros(3, *self.image_size, dtype=torch.float32)
            return zero_tensor, zero_tensor, stability_label, pair_id, stability

def get_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'image1': 'image'})

# 简化collate函数，只保留YOLO需要的数据
def custom_collate_fn(batch):
    clinic_rois, wood_rois, labels, pair_ids, stabilities = zip(*batch)
    return (torch.stack(clinic_rois, dim=0), torch.stack(wood_rois, dim=0),
            torch.tensor(labels, dtype=torch.long), list(pair_ids), list(stabilities))

def main():
    print("Starting prediction process...")
    
    # 检查必要文件（移除VMTL模型文件检查）
    required_files = [Config.yolo_model_path, 
                     Config.clinic_yolo_path, Config.wood_yolo_path, Config.csv_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"  ✗ {f}")
        return
    else:
        print("All required files found ✓")
    
    # 初始化ROI提取器
    try:
        roi_extractor = YOLOROIExtractor(Config.clinic_yolo_path, Config.wood_yolo_path, Config.device)
        print("✓ ROI extractor initialized")
    except Exception as e:
        print(f"✗ Failed to initialize ROI extractor: {e}")
        return
    

    # 加载YOLO模型
    try:
        yolo_model = YOLOConvNeXtV2Model(Config).to(Config.device)
        yolo_checkpoint = torch.load(Config.yolo_model_path, map_location=Config.device, weights_only=False)
        yolo_model.load_state_dict(yolo_checkpoint['model_state_dict'], strict=False)
        yolo_model.eval()
        print("✓ YOLO model loaded")
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return
    

    transform = get_transform(Config.img_size)
    dataset = VitiligoDataset(
        root_dir=Config.root_dir,
        csv_file=Config.csv_file,
        det_json_dir=Config.det_json_dir,
        seg_json_dir=Config.seg_json_dir,
        roi_extractor=roi_extractor,
        transform=transform,
        image_size=(Config.img_size, Config.img_size)
    )
    
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False, 
                           collate_fn=custom_collate_fn, num_workers=0)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    

    all_results = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Making predictions")
        for batch in pbar:
            clinic_roi, wood_roi, labels, pair_ids, stabilities = batch
            
            clinic_roi = clinic_roi.to(Config.device)
            wood_roi = wood_roi.to(Config.device)
            
            batch_size = clinic_roi.shape[0]
           
            # 只保留YOLO预测
            try:
                yolo_outputs = yolo_model(clinic_roi, wood_roi)
                yolo_logits = yolo_outputs['cls_pred_f']
                yolo_probs = F.softmax(yolo_logits, dim=1)
                yolo_preds = torch.argmax(yolo_probs, dim=1)
                yolo_confidences = yolo_probs.max(dim=1)[0]
            except Exception as e:
                print(f"YOLO prediction error: {e}")
                yolo_preds = torch.full((batch_size,), -1, dtype=torch.long, device=Config.device)
                yolo_confidences = torch.zeros(batch_size, device=Config.device)
            
           
            for i in range(batch_size):
                all_results.append({
                    'pair_id': pair_ids[i],
                    'stability': stabilities[i],
                    'true_label': labels[i].item(),
                    'yolo_prediction': yolo_preds[i].item(),
                    'yolo_confidence': yolo_confidences[i].item()
                })
            
           
            if all_results:
                last = all_results[-1]
                pbar.set_postfix({
                    'YOLO': last['yolo_prediction'],
                    'True': last['true_label']
                })
    

    print("Merging results with original data...")
    original_df = pd.read_csv(Config.csv_file)
    results_df = pd.DataFrame(all_results)
    
    merged_df = original_df.merge(
        results_df[['pair_id', 'stability', 'yolo_prediction', 'yolo_confidence']], 
        on=['pair_id', 'stability'], 
        how='left'
    )
    

    merged_df.to_csv(Config.output_csv, index=False)
    print(f"✓ Results saved to {Config.output_csv}")
    
 
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    total = len(all_results)
    valid_yolo = len([r for r in all_results if r['yolo_prediction'] != -1])
    
    print(f"Total samples: {total}")
    print(f"Valid YOLO predictions: {valid_yolo}")
    
    if valid_yolo > 0:
        yolo_correct = len([r for r in all_results 
                           if r['yolo_prediction'] != -1 and r['yolo_prediction'] == r['true_label']])
        print(f"YOLO Accuracy: {yolo_correct/valid_yolo:.4f} ({yolo_correct}/{valid_yolo})")
    
    print("="*60)
    print(f"Final CSV saved: {Config.output_csv}")

if __name__ == "__main__":
    main()
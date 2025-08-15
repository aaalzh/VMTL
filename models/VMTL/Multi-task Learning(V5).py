
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import logging
import warnings
from datetime import datetime
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
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
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "."
    csv_file = "../../../../datasets/data.csv"
    det_json_dir = "../../../../datasets/detection"
    seg_json_dir = "../../../../datasets/segmentation"
    lr = 1e-4
    weight_decay = 1e-5
    save_dir = "./training_results"
    log_dir = "./logs"
    encoder_name = "convnextv2_tiny"
    model_path = "training_results/final_staged_model.pth" 
    hidden_dim = 384
    num_classes = 2
    dropout_rate = 0.5
    detection_num_classes = 1
    use_amp = False 
    stage_epochs = [60, 40, 40, 40, 80, 100]
    stage_patience = [30, 25, 25, 25, 30, 50]  
    stage_lrs = [5e-5, 5e-5, 5e-5, 5e-5, 7e-5, 5e-5]  
    focal_alpha = 1
    focal_gamma = 2
    label_smoothing = 0.1
    
    gradient_tolerance = 0.5  
    max_grad_norm = 5.0  
    show_gradient_info = False
    
    data_expansion_factor = 3   

    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.save_dir, exist_ok=True)
        os.makedirs(cls.log_dir, exist_ok=True)
        return cls.save_dir, cls.log_dir

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir, log_dir = Config.create_dirs()
    log_file = os.path.join(Config.log_dir, f"training_{timestamp}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info("="*50)
    logger.info("Vitiligo Multi-Task Learning System Started - Continued Staged Training")
    logger.info("="*50)
    logger.info(f"Device: {Config.device}")
    logger.info(f"Image size: {Config.img_size}")
    logger.info(f"Batch size: {Config.batch_size}")
    logger.info("="*50)
    return logger


class ExpandedDataset(Dataset):
    
    def __init__(self, original_dataset, expansion_factor=0):

        self.original_dataset = original_dataset
        self.expansion_factor = expansion_factor
        
        
        if hasattr(original_dataset.dataset, 'image_size'):
            img_size = original_dataset.dataset.image_size[0]
        else:
            img_size = 224
        self.expansion_transform = get_expansion_transform(img_size)
        
      
        self.original_length = len(original_dataset)
        self.total_length = self.original_length * (1 + expansion_factor)
        
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        if idx < self.original_length:
            
            return self.original_dataset[idx]
        else:
            
            original_idx = (idx - self.original_length) % self.original_length
            
            
            original_transform = self.original_dataset.dataset.transform
            
           
            self.original_dataset.dataset.transform = self.expansion_transform
            
          
            data = self.original_dataset[original_idx]
            
           
            self.original_dataset.dataset.transform = original_transform
            
            return data

class VitiligoDataset(Dataset):
    def __init__(self, root_dir, csv_file, det_json_dir, seg_json_dir, transform=None, image_size=(224, 224)):
        self.data = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.det_json_dir = det_json_dir
        self.seg_json_dir = seg_json_dir
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
            return (torch.zeros(3, *self.image_size, dtype=torch.float32),
                    torch.zeros(3, *self.image_size, dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.long),
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros(1, *self.image_size, dtype=torch.float32),
                    torch.tensor(stability_label, dtype=torch.long))
        clinic_path = os.path.join(self.root_dir, clinic['image_path'].values[0])
        wood_path = os.path.join(self.root_dir, wood['image_path'].values[0])
        clinic_image = Image.open(clinic_path).convert('RGB').resize(self.image_size)
        wood_image = Image.open(wood_path).convert('RGB').resize(self.image_size)
        clinic_np = np.array(clinic_image)
        wood_np = np.array(wood_image)
        clinic_json = os.path.basename(clinic['image_path'].values[0]).rsplit('.', 1)[0] + '.json'
        wood_json = os.path.basename(wood['image_path'].values[0]).rsplit('.', 1)[0] + '.json'
        det_json_path = os.path.join(self.det_json_dir, clinic_json)
        seg_json_path = os.path.join(self.seg_json_dir, wood_json)
        bboxes = []
        class_ids = []
        if os.path.exists(det_json_path):
            with open(det_json_path, 'r', encoding='utf-8') as f:
                det_label = json.load(f)
            w, h = self.image_size
            for shape in det_label.get('shapes', []):
                if shape['shape_type'] != 'rectangle':
                    continue
                label = int(shape['label'])
                x1, y1 = shape['points'][0]
                x2, y2 = shape['points'][1]
                ow = det_label.get('imageWidth', clinic_image.size[0])
                oh = det_label.get('imageHeight', clinic_image.size[1])
                x1 = x1 * w / ow
                x2 = x2 * w / ow
                y1 = y1 * h / oh
                y2 = y2 * h / oh
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = abs(x2 - x1) / w
                bh = abs(y2 - y1) / h
                if bw > 0 and bh > 0 and 0 <= xc <= 1 and 0 <= yc <= 1:
                    bboxes.append([xc, yc, bw, bh])
                    class_ids.append(label)
        seg_mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.float32)
        if os.path.exists(seg_json_path):
            with open(seg_json_path, 'r', encoding='utf-8') as f:
                seg_label = json.load(f)
            for shape in seg_label.get('shapes', []):
                if shape['shape_type'] != 'polygon':
                    continue
                points = np.array(shape['points'], dtype=np.float32)
                ow = seg_label.get('imageWidth', wood_image.size[0])
                oh = seg_label.get('imageHeight', wood_image.size[1])
                points[:, 0] = points[:, 0] * self.image_size[0] / ow
                points[:, 1] = points[:, 1] * self.image_size[1] / oh
                points = points.astype(np.int32)
                if shape['label'] == '1':
                    cv2.fillPoly(seg_mask, [points], 1)
        if self.transform:
            augmented = self.transform(image=clinic_np, image1=wood_np, bboxes=bboxes, class_ids=class_ids, mask=seg_mask)
            clinic_img = augmented['image']
            wood_img = augmented['image1']
            bboxes = augmented['bboxes']
            class_ids = augmented['class_ids']
            seg_mask = augmented['mask']
        else:
            clinic_np = np.clip(clinic_np, 0, 255).astype(np.float32)
            wood_np = np.clip(wood_np, 0, 255).astype(np.float32)
            clinic_img = torch.tensor(clinic_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            wood_img = torch.tensor(wood_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            clinic_img = torch.clamp(clinic_img, 0.0, 1.0)
            wood_img = torch.clamp(wood_img, 0.0, 1.0)
        labels = torch.tensor(class_ids, dtype=torch.long) if class_ids else torch.zeros((0,), dtype=torch.long)
        det_target = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4), dtype=torch.float32)
        seg_target = (seg_mask.unsqueeze(0).to(dtype=torch.float32) if isinstance(seg_mask, torch.Tensor)
                      else torch.from_numpy(seg_mask).unsqueeze(0).to(dtype=torch.float32))
        return clinic_img, wood_img, labels, det_target, seg_target, torch.tensor(stability_label, dtype=torch.long)

def get_augmentation_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.4),
        A.GaussianBlur(3, p=0.3),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']), additional_targets={'image1': 'image'})

def get_expansion_transform(img_size=224):
    
    return A.Compose([
        A.Resize(img_size, img_size),
        
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.4),
        A.Rotate(limit=20, p=0.6),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        
  
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5),
        
      
        A.GaussianBlur(blur_limit=(3, 5), p=0.4),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        
       
        A.RandomShadow(p=0.3),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=0.2),
        
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']), additional_targets={'image1': 'image'})

def get_val_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']), additional_targets={'image1': 'image'})



def custom_collate_fn(batch):
    clinic_imgs, wood_imgs, labels_list, det_targets_list, seg_targets, stability_labels = zip(*batch)
    det_targets_list = [t if len(t) > 0 else torch.zeros((0, 4), dtype=torch.float32)
                       for t in det_targets_list]
    return (torch.stack(clinic_imgs, dim=0), torch.stack(wood_imgs, dim=0),
            list(labels_list), list(det_targets_list),
            torch.stack(seg_targets, dim=0), torch.stack(stability_labels, dim=0))

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

class MultiTaskModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = timm.create_model('convnextv2_tiny', pretrained=True, features_only=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        with torch.no_grad():
            dummy = torch.randn(1, 3, config.img_size, config.img_size)
            self.feature_dims = [f.shape[1] for f in self.backbone(dummy)]
        self.cross_attention = ConvQKVAttention(
            in_channels=self.feature_dims[-1],
            out_channels=self.feature_dims[-1],
            num_heads=8,
            num_iterations=3
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.feature_dims[-1], config.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.GELU()
        )
      
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        self.detection_head = nn.Sequential(
            nn.Conv2d(config.hidden_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 4, 1) 
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(config.hidden_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
            nn.Upsample(size=(config.img_size, config.img_size), mode='bilinear', align_corners=False)
        )
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(config.hidden_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, 1),
            nn.Upsample(size=(config.img_size, config.img_size), mode='bilinear', align_corners=False),
            nn.Sigmoid()
        )

        
        self.beta = nn.Parameter(torch.tensor(0.5))
        
        self.z_c_classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.z_w_classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.z_f_classifier = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, clinic_img, wood_img):
        clinic_feat = self.backbone(clinic_img)[-1]
        wood_feat = self.backbone(wood_img)[-1]
        fused_features, (clinic_attn, wood_attn), (z_c, z_w, z_f) = self.cross_attention(clinic_feat, wood_feat)
        fused = self.fusion_conv(fused_features)

      
        if z_c.dim() > 2:
            z_c = z_c.squeeze(-1).squeeze(-1)
        elif z_c.dim() == 1:
            z_c = z_c.unsqueeze(0)
        if z_w.dim() > 2:
            z_w = z_w.squeeze(-1).squeeze(-1)
        elif z_w.dim() == 1:
            z_w = z_w.unsqueeze(0)
        if z_f.dim() > 2:
            z_f = z_f.squeeze(-1).squeeze(-1)
        elif z_f.dim() == 1:
            z_f = z_f.unsqueeze(0)
     

        return {
            'classification': self.classifier(fused), 
            'detection': self.detection_head(fused),
            'segmentation': self.segmentation_head(fused),
            'reconstruction': self.reconstruction_head(fused),
            'attention_weights': (clinic_attn, wood_attn),
          
            'z_c': z_c,
            'z_w': z_w,
            'z_f': z_f
       
        }

class ContrastiveLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.pos_margin = nn.Parameter(torch.tensor(0.8, device=device)) 
        self.neg_margin = nn.Parameter(torch.tensor(0.3, device=device)) 
        
    def forward(self, z_c, z_w, class_labels):
        """
        Args:
            z_c: clinic features [batch_size, feature_dim]
            z_w: wood features [batch_size, feature_dim]
            class_labels: stability labels [batch_size] (0=stable, 1=progressive)
        """
        
        z_c_norm = F.normalize(z_c, p=2, dim=1)
        z_w_norm = F.normalize(z_w, p=2, dim=1)
        
        
        similarity = F.cosine_similarity(z_c_norm, z_w_norm, dim=1)
        
        
        stable_mask = (class_labels == 0)
        progressive_mask = (class_labels == 1)
        
        loss = 0.0
        
        
        if stable_mask.any():
            stable_similarities = similarity[stable_mask]
            stable_loss = torch.clamp(self.pos_margin - stable_similarities, min=0.0) ** 2
            loss += stable_loss.mean()
        
        
        if progressive_mask.any():
            progressive_similarities = similarity[progressive_mask]
            progressive_loss = torch.clamp(progressive_similarities - self.neg_margin, min=0.0) ** 2
            loss += progressive_loss.mean()
        
        return loss

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.1, ignore_index=-100, reduction='mean'):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        B, C = inputs.shape

        if self.label_smoothing > 0:
            smooth_targets = torch.full_like(inputs, self.label_smoothing / (C - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, C).float()

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        true_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * (1 - true_probs) ** self.gamma
        ce_loss_per_class = -smooth_targets * log_probs
        ce_loss_per_sample = ce_loss_per_class.sum(dim=1)
        focal_loss_per_sample = focal_weight * ce_loss_per_sample

        if self.reduction == 'mean':
            return focal_loss_per_sample.mean()
        elif self.reduction == 'sum':
            return focal_loss_per_sample.sum()
        else: 
            return focal_loss_per_sample



class StagedMultiTaskLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        focal_alpha = getattr(config, 'focal_alpha', 1)
        focal_gamma = getattr(config, 'focal_gamma', 2)
        label_smoothing = getattr(config, 'label_smoothing', 0.1)

        self.beta = nn.Parameter(torch.tensor(0.5).to(config.device))
        self.class_loss_focal = FocalLossWithLabelSmoothing(
            alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing
        )
        self.det_loss = nn.MSELoss()
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.recon_loss = nn.MSELoss()
        self.det_empty_weight = 0.1
        self.contrastive_loss = ContrastiveLoss(config.device)

        self.log_vars = nn.ParameterDict({
            'log_var_det': nn.Parameter(torch.zeros(1).to(config.device)),
            'log_var_seg': nn.Parameter(torch.zeros(1).to(config.device)),
            'log_var_recon': nn.Parameter(torch.zeros(1).to(config.device)),
            'log_var_class': nn.Parameter(torch.zeros(1).to(config.device)),
            'log_var_con': nn.Parameter(torch.zeros(1).to(config.device)) # For contrastive loss
        })
     

    def forward(self, outputs, targets, stage):
        clinic_img, wood_img, det_labels, det_targets, seg_targets, class_labels = targets
        losses = {}
        total_loss = 0.0

        if stage == 0: 
           
            losses['classification'] = self.class_loss_focal(outputs['classification'], class_labels)

           
            det_loss = []
            for i, dt in enumerate(det_targets):
                pred = outputs['detection'][i]
                if len(dt) > 0:
                    tgt = dt.to(outputs['detection'].device)
                    if pred.shape[1] >= 4:
                        pred_boxes = pred[:4]
                        if pred_boxes.dim() > 1:
                            pred_boxes = pred_boxes.mean(dim=(1, 2))
                        if tgt.dim() == 2 and tgt.shape[1] == 4:
                            tgt = tgt.mean(dim=0)
                        elif tgt.dim() == 1 and tgt.shape[0] == 4:
                            pass
                        else:
                            det_loss.append(torch.tensor(0.0, device=pred.device))
                            continue
                        det_loss.append(self.det_loss(pred_boxes, tgt))
                    else:
                        det_loss.append(torch.tensor(0.0, device=pred.device))
                else:
                    if pred.shape[1] >= 4:
                        pred_boxes = pred[:4]
                        if pred_boxes.dim() > 1:
                            pred_boxes = pred_boxes.mean(dim=(1, 2))
                        default_box = torch.tensor([0.5, 0.5, 0.0, 0.0], device=pred.device)
                        det_loss.append(self.det_loss(pred_boxes, default_box))
                    else:
                        det_loss.append(torch.tensor(0.0, device=pred.device))

            if len(det_loss) > 0:
                losses['detection'] = torch.stack(det_loss).mean() * (
                    self.det_empty_weight if all(len(dt) == 0 for dt in det_targets) else 1.0)
            else:
                losses['detection'] = torch.tensor(0.0, device=outputs['detection'].device)

           
            losses['segmentation'] = self.seg_loss(outputs['segmentation'], seg_targets)

            
            recon_loss = 0.0
            for i in range(clinic_img.shape[0]):
                recon_loss += self.recon_loss(outputs['reconstruction'][i], clinic_img[i])
                recon_loss += self.recon_loss(outputs['reconstruction'][i], wood_img[i])
            losses['reconstruction'] = recon_loss / (2 * clinic_img.shape[0])

           
            losses['contrastive'] = self.contrastive_loss(outputs['z_c'], outputs['z_w'], class_labels)

            
            task_losses = {
                'detection': losses['detection'],
                'segmentation': losses['segmentation'],
                'reconstruction': losses['reconstruction'],
                'classification': losses['classification'],
                'contrastive': losses['contrastive']
            }

            for task_name, task_loss in task_losses.items():
                log_var_key = f'log_var_{task_name[:3]}'  # 'det', 'seg', 'rec', 'cla', 'con'
                if log_var_key in self.log_vars:
                    precision = torch.exp(-self.log_vars[log_var_key])
                    total_loss += precision * task_loss + 0.5 * self.log_vars[log_var_key]
                else:
                    total_loss += task_loss

            losses['total'] = total_loss
            return losses

        elif stage == 1:  
            recon_loss = 0.0
            for i in range(clinic_img.shape[0]):
                recon_loss += self.recon_loss(outputs['reconstruction'][i].to(Config.device), clinic_img[i].to(Config.device))
                recon_loss += self.recon_loss(outputs['reconstruction'][i].to(Config.device), wood_img[i].to(Config.device))
            losses['reconstruction'] = recon_loss / (2 * clinic_img.shape[0])
            
            precision = torch.exp(-self.log_vars['log_var_recon'])
            total_loss += precision * losses['reconstruction'].to(Config.device) + 0.5 * self.log_vars['log_var_recon']
            losses['total'] = total_loss
            return losses

        elif stage == 2:  
            det_loss = []
            for i, dt in enumerate(det_targets):
                pred = outputs['detection'][i]
                if len(dt) > 0:
                    tgt = dt.to(outputs['detection'].device)
                    if pred.shape[1] >= 4: 
                        pred_boxes = pred[:4] 
                        if pred_boxes.dim() > 1:
                             pred_boxes = pred_boxes.mean(dim=(1, 2))
                        if tgt.dim() == 2 and tgt.shape[1] == 4:
                            tgt = tgt.mean(dim=0)
                        elif tgt.dim() == 1 and tgt.shape[0] == 4:
                             pass 
                        else:
                            det_loss.append(torch.tensor(0.0, device=pred.device))
                            continue
                        det_loss.append(self.det_loss(pred_boxes, tgt))
                    else:
                        det_loss.append(torch.tensor(0.0, device=pred.device))
                else:
                    if pred.shape[1] >= 4:
                        pred_boxes = pred[:4]
                        if pred_boxes.dim() > 1:
                            pred_boxes = pred_boxes.mean(dim=(1, 2))
                        default_box = torch.tensor([0.5, 0.5, 0.0, 0.0], device=pred.device)
                        det_loss.append(self.det_loss(pred_boxes, default_box))
                    else:
                        det_loss.append(torch.tensor(0.0, device=pred.device))

            if len(det_loss) > 0:
                losses['detection'] = torch.stack(det_loss).mean() * (
                    self.det_empty_weight if all(len(dt) == 0 for dt in det_targets) else 1.0)
            else:
                losses['detection'] = torch.tensor(0.0, device=outputs['detection'].device)
            
            precision = torch.exp(-self.log_vars['log_var_det'])
            total_loss += precision * losses['detection'] + 0.5 * self.log_vars['log_var_det']
            losses['total'] = total_loss
            return losses

        elif stage == 3:  
            losses['segmentation'] = self.seg_loss(outputs['segmentation'], seg_targets)
            
            precision = torch.exp(-self.log_vars['log_var_seg'])
            total_loss += precision * losses['segmentation'] + 0.5 * self.log_vars['log_var_seg']
            losses['total'] = total_loss
            return losses

        elif stage == 4: 
            losses['classification'] = self.class_loss_focal(outputs['classification'], class_labels)
            losses['contrastive'] = self.contrastive_loss(outputs['z_c'], outputs['z_w'], class_labels)
            
          
            precision_class = torch.exp(-self.log_vars['log_var_class'])
            precision_contrast = torch.exp(-self.log_vars['log_var_con'])
            total_loss += precision_class * losses['classification'] + 0.5 * self.log_vars['log_var_class']
            total_loss += precision_contrast * losses['contrastive'] + 0.5 * self.log_vars['log_var_con']
            losses['total'] = total_loss
            return losses

        elif stage == 5:  
            
            losses['classification'] = self.class_loss_focal(outputs['classification'], class_labels)

           
            det_loss = []
            for i, dt in enumerate(det_targets):
                pred = outputs['detection'][i]
                if len(dt) > 0:
                    tgt = dt.to(outputs['detection'].device)
                    if pred.shape[1] >= 4:
                        pred_boxes = pred[:4]
                        if pred_boxes.dim() > 1:
                            pred_boxes = pred_boxes.mean(dim=(1, 2))
                        if tgt.dim() == 2 and tgt.shape[1] == 4:
                            tgt = tgt.mean(dim=0)
                        elif tgt.dim() == 1 and tgt.shape[0] == 4:
                            pass
                        else:
                            det_loss.append(torch.tensor(0.0, device=pred.device))
                            continue
                        det_loss.append(self.det_loss(pred_boxes, tgt))
                    else:
                        det_loss.append(torch.tensor(0.0, device=pred.device))
                else:
                    if pred.shape[1] >= 4:
                        pred_boxes = pred[:4]
                        if pred_boxes.dim() > 1:
                            pred_boxes = pred_boxes.mean(dim=(1, 2))
                        default_box = torch.tensor([0.5, 0.5, 0.0, 0.0], device=pred.device)
                        det_loss.append(self.det_loss(pred_boxes, default_box))
                    else:
                        det_loss.append(torch.tensor(0.0, device=pred.device))

            if len(det_loss) > 0:
                losses['detection'] = torch.stack(det_loss).mean() * (
                    self.det_empty_weight if all(len(dt) == 0 for dt in det_targets) else 1.0)
            else:
                losses['detection'] = torch.tensor(0.0, device=outputs['detection'].device)

          
            losses['segmentation'] = self.seg_loss(outputs['segmentation'], seg_targets)

          
            recon_loss = 0.0
            for i in range(clinic_img.shape[0]):
                recon_loss += self.recon_loss(outputs['reconstruction'][i], clinic_img[i])
                recon_loss += self.recon_loss(outputs['reconstruction'][i], wood_img[i])
            losses['reconstruction'] = recon_loss / (2 * clinic_img.shape[0])

         
            losses['contrastive'] = self.contrastive_loss(outputs['z_c'], outputs['z_w'], class_labels)

           
            task_losses = {
                'detection': losses['detection'],
                'segmentation': losses['segmentation'],
                'reconstruction': losses['reconstruction'],
                'classification': losses['classification'],
                'contrastive': losses['contrastive']
            }

            for task_name, task_loss in task_losses.items():
                log_var_key = f'log_var_{task_name[:3]}'  # 'det', 'seg', 'rec', 'cla', 'con'
                if log_var_key in self.log_vars:
                    precision = torch.exp(-self.log_vars[log_var_key])
                    total_loss += precision * task_loss + 0.5 * self.log_vars[log_var_key]
                else:
                    total_loss += task_loss

            losses['total'] = total_loss
            return losses

        else:
            raise ValueError(f"Invalid stage: {stage}")

def calculate_metrics(y_true, y_pred, y_prob=None):
    if len(y_true) == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'specificity': 0, 'auc': 0, 'accuracy': 0}

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0

    metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'specificity': specificity, 'accuracy': accuracy}

    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except:
            metrics['auc'] = 0
    else:
        metrics['auc'] = 0

    return metrics

def train_epoch(model, dataloader, criterion, optimizer, device, logger, stage, scaler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    pbar = tqdm(dataloader, desc=f"Training Stage {stage}")
    use_amp = Config.use_amp and scaler is not None

    for clinic_img, wood_img, det_labels, det_targets, seg_targets, class_labels in pbar:
        clinic_img = clinic_img.to(device)
        wood_img = wood_img.to(device)
        seg_targets = seg_targets.to(device)
        class_labels = class_labels.to(device)

        optimizer.zero_grad()

        
        if use_amp:
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(clinic_img, wood_img)
                losses = criterion(outputs, (clinic_img, wood_img, det_labels, det_targets, seg_targets, class_labels), stage)
                loss = losses['total']
        else:
            outputs = model(clinic_img, wood_img)
            losses = criterion(outputs, (clinic_img, wood_img, det_labels, det_targets, seg_targets, class_labels), stage)
            loss = losses['total']

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            optimizer.zero_grad()
            logger.warning(f"Stage {stage} NaN/Inf loss, Skipping")
            continue

       
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        loss_params = [p for p in criterion.log_vars.parameters() if p.requires_grad]
        contrastive_params = [p for p in criterion.contrastive_loss.parameters() if p.requires_grad]
        all_trainable_params = trainable_params + loss_params + contrastive_params
        
        if len(all_trainable_params) == 0:
            logger.warning(f"Stage {stage} - No trainable parameters, skipping update")
            optimizer.zero_grad()
            continue
        
      
        nan_count = 0
        inf_count = 0
        total_params = 0
        
        
        for p in all_trainable_params:
            if p.grad is not None:
                total_params += 1
                
                if torch.isnan(p.grad).any():
                    nan_count += 1
                   
                    p.grad[torch.isnan(p.grad)] = 0.0
               
                if torch.isinf(p.grad).any():
                    inf_count += 1
                    
                    p.grad = torch.clamp(p.grad, -1e6, 1e6)
        
       
        if total_params == 0:
            logger.warning(f"Stage {stage} - No gradients available, skipping update")
            optimizer.zero_grad()
            continue
        elif (nan_count + inf_count) > total_params * Config.gradient_tolerance:
            logger.warning(f"Stage {stage} - Too many invalid gradients (NaN: {nan_count}, Inf: {inf_count} out of {total_params}), skipping update")
            optimizer.zero_grad()
            continue
        elif nan_count > 0 or inf_count > 0:
            if Config.show_gradient_info:
                logger.info(f"Stage {stage} - Fixed gradients (NaN: {nan_count}, Inf: {inf_count} out of {total_params})")
        
        if Config.show_gradient_info:
            log_var_grad_info = []
            for key, param in criterion.log_vars.items():
                if param.grad is not None:
                    log_var_grad_info.append(f"{key}: {param.grad.item():.6f}")
                else:
                    log_var_grad_info.append(f"{key}: no_grad")
            if log_var_grad_info:
                logger.info(f"Stage {stage} - Log_var gradients: {', '.join(log_var_grad_info)}")
            
            contrastive_grad_info = []
            for name, param in criterion.contrastive_loss.named_parameters():
                if param.grad is not None:
                    contrastive_grad_info.append(f"{name}: {param.grad.item():.6f}")
                else:
                    contrastive_grad_info.append(f"{name}: no_grad")
            if contrastive_grad_info:
                logger.info(f"Stage {stage} - Contrastive gradients: {', '.join(contrastive_grad_info)}")
        
        
        grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=Config.max_grad_norm)
        if grad_norm > Config.max_grad_norm * 4 and Config.show_gradient_info: 
            logger.info(f"Stage {stage} - Large gradient norm clipped: {grad_norm:.4f}")
        
        
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()

        total_loss += loss.item()


       
        try:
            with torch.no_grad():
               
                classification_logits = outputs['classification']
                probs = F.softmax(classification_logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(class_labels.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
        except Exception as e:
            logger.warning(f"Error calculating metrics in stage {stage}: {e}")
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)

    if all_preds and all_labels:
        metrics = calculate_metrics(all_labels, all_preds, np.array(all_probs) if all_probs else None)
        
     
        if metrics.get('specificity', 0) == 0 and metrics.get('precision', 0) == 1 and metrics.get('recall', 0) == 1:
            metrics['specificity'] = 1e-6  
            
        return avg_loss, metrics
    else:
        return avg_loss, {}

def validate_epoch(model, dataloader, criterion, device, logger, stage):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    pbar = tqdm(dataloader, desc=f"Validation Stage {stage}")

    with torch.no_grad():
        for clinic_img, wood_img, det_labels, det_targets, seg_targets, class_labels in pbar:
            clinic_img = clinic_img.to(device)
            wood_img = wood_img.to(device)
            seg_targets = seg_targets.to(device)
            class_labels = class_labels.to(device)

            outputs = model(clinic_img, wood_img)
            losses = criterion(outputs, (clinic_img, wood_img, det_labels, det_targets, seg_targets, class_labels), stage)
            loss = losses['total']
            total_loss += loss.item()

            
            
            try:
               
                classification_logits = outputs['classification']
                probs = F.softmax(classification_logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(class_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            except Exception as e:
                logger.warning(f"Error calculating metrics in stage {stage}: {e}")

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)

    if all_preds and all_labels:
        return avg_loss, calculate_metrics(all_labels, all_preds, np.array(all_probs) if all_probs else None)
    else:
        return avg_loss, {}

def freeze_model_parts(model, stage):

    for param in model.parameters():
        param.requires_grad = True

    if stage == 0:  
  
        pass
    elif stage == 1: 
        for name, child in model.named_children():
            if name not in ['backbone', 'cross_attention', 'fusion_conv', 'reconstruction_head']:
                for param in child.parameters():
                    param.requires_grad = False
    elif stage == 2:  
        for name, child in model.named_children():
            if name not in ['backbone', 'cross_attention', 'fusion_conv', 'detection_head']:
                for param in child.parameters():
                    param.requires_grad = False
    elif stage == 3:  
        for name, child in model.named_children():
            if name not in ['backbone', 'cross_attention', 'fusion_conv', 'segmentation_head']:
                for param in child.parameters():
                    param.requires_grad = False
    elif stage == 4:  
        for name, child in model.named_children():
            if name not in ['backbone', 'cross_attention', 'fusion_conv', 'classifier']:
                for param in child.parameters():
                    param.requires_grad = False
   


def calculate_psr_score(metrics):
    return metrics.get('precision', 0) * metrics.get('specificity', 0) * metrics.get('recall', 0)

def main():
    logger = setup_logging()

    model_path = Config.model_path
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} not found!")
        return

    checkpoint = torch.load(model_path, map_location=Config.device, weights_only=False)
    model_config = checkpoint.get('config', {})

    for key, value in model_config.items():
        if hasattr(Config, key):
            setattr(Config, key, value)

    model = MultiTaskModel(Config).to(Config.device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logger.info(f"Loaded model from {model_path}")
    
    
 
    data = pd.read_csv(Config.csv_file)
    grouped = data.groupby(['pair_id', 'stability'])
    group_keys = list(grouped.groups.keys())
    train_keys, val_keys = train_test_split(group_keys, test_size=0.2, random_state=42, stratify=[k[1] for k in group_keys])


    full_dataset = VitiligoDataset(root_dir=Config.root_dir, csv_file=Config.csv_file,
                                   det_json_dir=Config.det_json_dir, seg_json_dir=Config.seg_json_dir,
                                   transform=None, image_size=(Config.img_size, Config.img_size))
    
   
    train_indices = [i for i, k in enumerate(full_dataset.group_keys) if k in train_keys]
    val_indices = [i for i, k in enumerate(full_dataset.group_keys) if k in val_keys]
    

    original_train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    
    original_train_dataset.dataset.transform = get_augmentation_transform(Config.img_size)
    val_dataset.dataset.transform = get_val_transform(Config.img_size)
    
    
    if Config.data_expansion_factor > 0:
        train_dataset = ExpandedDataset(original_train_dataset, Config.data_expansion_factor)
        logger.info(f"Data expansion enabled: {Config.data_expansion_factor}x")
        logger.info(f"Original training set size: {len(original_train_dataset)}")
        logger.info(f"Expanded training set size: {len(train_dataset)}")
    else:
        train_dataset = original_train_dataset
        logger.info("Data expansion disabled")

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

   
    freeze_model_parts(model, 4)  
    model.eval()
    _, init_metrics = validate_epoch(model, val_loader, StagedMultiTaskLoss(Config), Config.device, logger, 4)
    best_psr = calculate_psr_score(init_metrics)
    best_precision = init_metrics.get('precision', 0)
    best_recall = init_metrics.get('recall', 0) 
    best_specificity = init_metrics.get('specificity', 0)
    logger.info(f"Initial P*S*R score: {best_psr:.6f}")
    
    
    for param in model.parameters():
        param.requires_grad = True
    
  
    def get_best_model_path(precision, specificity, recall):
        return os.path.join(Config.save_dir, f'best_model_P{precision:.3f}_S{specificity:.3f}_R{recall:.3f}.pth')
    
    def remove_old_best_model(old_path):
   
        if old_path and os.path.exists(old_path):
            try:
                os.remove(old_path)
                logger.info(f"Removed old best model: {old_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old model {old_path}: {e}")
    
   
    current_best_model_path = get_best_model_path(best_precision, best_specificity, best_recall)
    torch.save({
        'model_state_dict': model.state_dict(),
        'psr_score': best_psr,
        'precision': best_precision,
        'recall': best_recall,
        'specificity': best_specificity,
        'config': {
            'img_size': Config.img_size,
            'batch_size': Config.batch_size,
            'hidden_dim': Config.hidden_dim,
            'num_classes': Config.num_classes,
            'dropout_rate': Config.dropout_rate,
            'detection_num_classes': Config.detection_num_classes,
            'focal_alpha': Config.focal_alpha,
            'focal_gamma': Config.focal_gamma,
            'label_smoothing': Config.label_smoothing
        }
    }, current_best_model_path)
    logger.info(f"Initial best model saved: {current_best_model_path}")



    criterion = StagedMultiTaskLoss(Config).to(Config.device)
    scaler = GradScaler(enabled=Config.use_amp)

    num_stages = 6
    logger.info(f"Starting training with {num_stages} stages (0 to {num_stages-1})")
    
    for stage in range(num_stages):
        logger.info(f"\n{'='*30} Starting Training Stage {stage} {'='*30}")

        
        if os.path.exists(current_best_model_path):
            try:
                checkpoint = torch.load(current_best_model_path, map_location=Config.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info(f"Stage {stage}: Loaded global best model (P*S*R: {checkpoint.get('psr_score', 0):.6f}): {current_best_model_path}")

                criterion = StagedMultiTaskLoss(Config).to(Config.device)
                logger.info(f"Stage {stage}: Recreated criterion with fresh log_var parameters")
            except Exception as e:
                logger.warning(f"Stage {stage}: Failed to load global best model: {e}. Using current model state.")

        freeze_model_parts(model, stage)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Stage {stage}: Trainable params: {trainable_params}/{total_params}")

        
        model_params = [p for p in model.parameters() if p.requires_grad]
        loss_params = [p for p in criterion.log_vars.parameters() if p.requires_grad]
        contrastive_params = [p for p in criterion.contrastive_loss.parameters() if p.requires_grad]
        
        logger.info(f"Stage {stage}: Model params: {len(model_params)}, Loss params: {len(loss_params)}, Contrastive params: {len(contrastive_params)}")

        if len(loss_params) == 0:
            logger.warning(f"Stage {stage}: No trainable log_var parameters found!")
            for param in criterion.log_vars.parameters():
                param.requires_grad = True
            loss_params = [p for p in criterion.log_vars.parameters() if p.requires_grad]
            logger.info(f"Stage {stage}: After forcing requires_grad=True, Loss params: {len(loss_params)}")
        
        if len(contrastive_params) == 0:
            logger.warning(f"Stage {stage}: No trainable contrastive parameters found!")

            for param in criterion.contrastive_loss.parameters():
                param.requires_grad = True
            contrastive_params = [p for p in criterion.contrastive_loss.parameters() if p.requires_grad]
            logger.info(f"Stage {stage}: After forcing requires_grad=True, Contrastive params: {len(contrastive_params)}")
        
        param_groups = [
            {'params': model_params, 'lr': Config.stage_lrs[stage]},
            {'params': loss_params, 'lr': Config.stage_lrs[stage] * 0.1},
            {'params': contrastive_params, 'lr': Config.stage_lrs[stage] * 0.05}  
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=Config.weight_decay)


        T_max = Config.stage_epochs[stage]

        warmup_epochs = min(5, T_max // 10) 

        def lr_lambda_cosine_with_warmup(epoch):
            if epoch < warmup_epochs:
               
                return float(epoch + 1) / float(max(1, warmup_epochs))
            else:
              
                cycle_epoch = epoch - warmup_epochs
                return 0.5 * (1 + np.cos(np.pi * cycle_epoch / max(1, T_max - warmup_epochs)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_cosine_with_warmup)
       

       
        stage_best_score = -float('inf')
        patience_counter = 0 

        for epoch in range(Config.stage_epochs[stage]):
            logger.info(f"\nStage {stage} - Epoch {epoch+1}/{Config.stage_epochs[stage]}")

            train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, Config.device, logger, stage, scaler)
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, Config.device, logger, stage)

            
            val_precision = val_metrics.get('precision', 0)
            val_recall = val_metrics.get('recall', 0)
            val_specificity = val_metrics.get('specificity', 0)
            current_psr = calculate_psr_score(val_metrics)
            current_score = current_psr
            
            logger.info(f"Stage {stage} - Epoch {epoch+1}")
            logger.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val Specificity: {val_specificity:.4f}")
            logger.info(f"  Val F1: {val_metrics.get('f1', 0):.4f} | Val AUC: {val_metrics.get('auc', 0):.4f} | Val Accuracy: {val_metrics.get('accuracy', 0):.4f}")
            logger.info(f"  Combined Score (P*S*R): {current_psr:.6f}")
            
           
            if stage in [0, 4, 5]:
           
                try:
                    with torch.no_grad():
                        model.eval()
                        sample_batch = next(iter(val_loader))
                        clinic_img, wood_img, det_labels, det_targets, seg_targets, class_labels = sample_batch
                        clinic_img = clinic_img.to(Config.device)
                        wood_img = wood_img.to(Config.device)
                        seg_targets = seg_targets.to(Config.device)
                        class_labels = class_labels.to(Config.device)
                        
                        outputs = model(clinic_img, wood_img)
                        losses = criterion(outputs, (clinic_img, wood_img, det_labels, det_targets, seg_targets, class_labels), stage)
                        
                    
                        if 'classification' in losses:
                            logger.info(f"    Classification Loss: {losses['classification'].item():.6f}")
                        if 'detection' in losses:
                            logger.info(f"    Detection Loss: {losses['detection'].item():.6f}")
                        if 'segmentation' in losses:
                            logger.info(f"    Segmentation Loss: {losses['segmentation'].item():.6f}")
                        if 'reconstruction' in losses:
                            logger.info(f"    Reconstruction Loss: {losses['reconstruction'].item():.6f}")
                        if 'contrastive' in losses:
                            logger.info(f"    Contrastive Loss: {losses['contrastive'].item():.6f}")
                        
                      
                        logger.info("    Dynamic Weights (exp(-log_var)):")
                        for key, log_var in criterion.log_vars.items():
                            weight = torch.exp(-log_var).item()
                            grad_info = f" (grad: {log_var.grad.item():.6f})" if log_var.grad is not None else " (no grad)"
                            logger.info(f"      {key}: {weight:.6f}{grad_info}")
                        
                        
                        pos_margin = criterion.contrastive_loss.pos_margin.item()
                        neg_margin = criterion.contrastive_loss.neg_margin.item()
                        logger.info(f"    Contrastive margins - Pos: {pos_margin:.4f} | Neg: {neg_margin:.4f}")
                        
                        model.train()
                except Exception as e:
                    logger.warning(f"Failed to log detailed losses: {e}")
                   
                    try:
                        pos_margin = criterion.contrastive_loss.pos_margin.item()
                        neg_margin = criterion.contrastive_loss.neg_margin.item()
                        logger.info(f"  Contrastive margins - Pos: {pos_margin:.4f} | Neg: {neg_margin:.4f}")
                    except:
                        pass
            
           
            
            if current_psr > best_psr:
               
                remove_old_best_model(current_best_model_path)
                
                
                best_psr = current_psr
                best_precision = val_precision
                best_recall = val_recall
                best_specificity = val_specificity
                
                
                current_best_model_path = get_best_model_path(best_precision, best_specificity, best_recall)
                
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psr_score': best_psr,
                    'precision': best_precision,
                    'recall': best_recall,
                    'specificity': best_specificity,
                    'stage': stage,
                    'epoch': epoch,
                    'config': {
                        'img_size': Config.img_size,
                        'batch_size': Config.batch_size,
                        'hidden_dim': Config.hidden_dim,
                        'num_classes': Config.num_classes,
                        'dropout_rate': Config.dropout_rate,
                        'detection_num_classes': Config.detection_num_classes,
                        'focal_alpha': Config.focal_alpha,
                        'focal_gamma': Config.focal_gamma,
                        'label_smoothing': Config.label_smoothing
                    }
                }, current_best_model_path)
                logger.info(f"Stage {stage} - New global best model saved: {current_best_model_path} (P*S*R: {best_psr:.6f})")
            
            
            if current_score > stage_best_score:
                stage_best_score = current_score
                patience_counter = 0
                logger.info(f"Stage {stage} - Stage best score updated: {stage_best_score:.6f}")
            else:
                patience_counter += 1

            scheduler.step()

            if patience_counter >= Config.stage_patience[stage]:
                logger.info(f"Stage {stage} - Early stopping triggered. Best P*S*R: {stage_best_score:.6f}")
                break

        logger.info(f"Completed Stage {stage}. Current global best P*S*R: {best_psr:.6f}")
        logger.info(f"Next stage will load global best model: {current_best_model_path}")

    logger.info(f"All {num_stages} stages completed successfully!")
    logger.info(f"Final global best model: {current_best_model_path}")
    logger.info(f"Final P*S*R score: {best_psr:.6f} (P: {best_precision:.3f}, S: {best_specificity:.3f}, R: {best_recall:.3f})")
    
   
    if os.path.exists(current_best_model_path):
        logger.info(f"Training completed! Best model saved as: {current_best_model_path}")
    else:
        logger.warning("Training completed, but no best model file found.")

if __name__ == "__main__":
    main()



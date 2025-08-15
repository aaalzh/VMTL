from logging import config
import os
from pickletools import optimize
from pyexpat import model
import random
import json
from venv import logger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from timm.models.convnext import ConvNeXtBlock
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score
import seaborn as sns
import cv2
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
class Config:
    img_size = 224
    roi_size = 224
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "./"
    csv_file = "../datasets/data.csv"
    clinic_yolo_path = "../outputs/YOLO/checkpoints/detection/v12best.pt"
    wood_yolo_path = "../outputs/YOLO/checkpoints/segmentation/v12best.pt"

    lr = 7e-5
    weight_decay = 1e-4
    num_epochs = 500
    patience = 70
    save_dir = "./yolo_training_resultsv86"
    log_dir = "./yolo_logsv86"   
    encoder_name = "convnextv2_tiny"
    hidden_dim = 384
    num_classes = 2
    dropout_rate = 0.5
    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.save_dir, exist_ok=True)
        os.makedirs(cls.log_dir, exist_ok=True)
        return cls.save_dir, cls.log_dir
    @classmethod
    def get_model_save_path(cls):
        return os.path.join(cls.save_dir, "best_yolo_convnext_model.pth")
_logger_initialized = False
def setup_logging():
    global _logger_initialized
    if _logger_initialized:
        return logging.getLogger(__name__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir, log_dir = Config.create_dirs()
    log_file = os.path.join(Config.log_dir, f"yolo_convnext_training_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    _logger_initialized = True
def setup_logging():
    global _logger_initialized
    if _logger_initialized:
        return logging.getLogger(__name__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir, log_dir = Config.create_dirs()
    log_file = os.path.join(Config.log_dir, f"yolo_convnext_training_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    _logger_initialized = True
    logger.info(f"YOLO-ConvNeXtV2 Training log: {log_file}")
    logger.info(f"Device: {Config.device}")
    return logger

def get_logger():
    return logging.getLogger(__name__)

class YOLOROIExtractor:
    def __init__(self, clinic_model_path, wood_model_path, device):
        self.device = device
        self.clinic_model = YOLO(clinic_model_path)
        self.wood_model = YOLO(wood_model_path)
        get_logger().info(f"Loaded clinic detection model from: {clinic_model_path}")
        get_logger().info(f"Loaded wood segmentation model from: {wood_model_path}")

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

class YOLOVitiligoDataset(Dataset):
    def __init__(self, root_dir, csv_file, roi_extractor, transform=None, image_size=(224, 224)):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.roi_extractor = roi_extractor
        self.image_size = image_size
        self.transform = transform if transform is not None else A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ], additional_targets={'image1': 'image'})
        self.grouped = self.data.groupby(['pair_id', 'stability'])
        self.group_keys = list(self.grouped.groups.keys())
        get_logger().info(f"YOLO Dataset initialized with {len(self.group_keys)} sample pairs")

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        pair_id, stability = self.group_keys[idx]
        pairs = self.grouped.get_group((pair_id, stability))
        stability_label = 0 if stability.lower() == 'stable' else 1
        clinic = pairs[pairs['image_type'] == 'clinic']
        wood = pairs[pairs['image_type'] == 'wood']
        if clinic.empty or wood.empty:
            get_logger().warning(f"Missing clinic or wood image for pair_id {pair_id}, stability {stability}")
            return (torch.zeros(3, *self.image_size, dtype=torch.float32),
                    torch.zeros(3, *self.image_size, dtype=torch.float32),
                    torch.tensor(0, dtype=torch.long))
        clinic_path = os.path.join(self.root_dir, clinic['image_path'].values[0])
        wood_path = os.path.join(self.root_dir, wood['image_path'].values[0])
        clinic_image = cv2.imread(clinic_path)
        clinic_image = cv2.cvtColor(clinic_image, cv2.COLOR_BGR2RGB)
        wood_image = cv2.imread(wood_path)
        wood_image = cv2.cvtColor(wood_image, cv2.COLOR_BGR2RGB)
        try:
            clinic_roi = self.roi_extractor.extract_clinic_roi(clinic_image)
            wood_roi = self.roi_extractor.extract_wood_roi(wood_image)
        except Exception as e:
            get_logger().warning(f"YOLO ROI extraction failed for {pair_id}: {e}")
            clinic_roi = clinic_image
            wood_roi = wood_image
        clinic_roi = cv2.resize(clinic_roi, self.image_size)
        wood_roi = cv2.resize(wood_roi, self.image_size)
        if self.transform:
            augmented = self.transform(image=clinic_roi, image1=wood_roi)
            clinic_tensor = augmented['image']
            wood_tensor = augmented['image1']
        else:
            clinic_tensor = torch.tensor(clinic_roi.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            wood_tensor = torch.tensor(wood_roi.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        stability_label = torch.tensor(stability_label, dtype=torch.long)
        return clinic_tensor, wood_tensor, stability_label

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
        head_dim = C // self.num_heads
        q = self.q_conv(x_aux).view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        k = self.k_conv(x_main).view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        v = self.v_conv(x_main).view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        z = torch.matmul(attention_weights, v)
        z = z.transpose(2, 3).contiguous().view(B, C, H, W)
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
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.1, num_classes=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()
class YOLOConvNeXtV2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = timm.create_model(
            config.encoder_name,
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
        self.head_c = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        self.head_w = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
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
        cls_pred_c = self.head_c(z_c)
        cls_pred_w = self.head_w(z_w)
        cls_pred_f = self.head_f(z_f)
        return {
            'cls_pred_c': cls_pred_c,
            'cls_pred_w': cls_pred_w,
            'cls_pred_f': cls_pred_f,
            'z_c': z_c,
            'z_w': z_w,
            'z_f': z_f,
            'fused_features': fused_features,
            'clinic_attention': clinic_attention,
            'wood_attention': wood_attention
        }
    def cross_entropy_loss(self, cls_pred, labels, smoothing=0.1):
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        return criterion(cls_pred, labels)
    def compute_losses(self, outputs, stability_labels, loss_weights):
        focal = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1, num_classes=self.config.num_classes)
        cls_loss = focal(outputs['cls_pred_f'], stability_labels)
        losses = {'classification': cls_loss}
        total_loss = cls_loss
        losses['total'] = total_loss.item()
        return losses, total_loss
def compute_metrics(cls_pred, labels):
    if cls_pred is None or labels is None or len(labels) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'f1': 0.0,
            'score': 0.0,
            'accuracy': 0.0,
            'auc': 0.0,
            'confusion_matrix': np.zeros((2, 2))
        }
    preds = torch.argmax(cls_pred, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    try:
        cm = confusion_matrix(labels, preds, labels=[0, 1])
    except:
        cm = np.zeros((2, 2))
    if cm.shape != (2, 2):
        cm = np.zeros((2, 2))
    tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
    fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
    tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    score = precision * specificity * recall
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    try:
        if len(np.unique(labels)) > 1:
            probs = torch.softmax(cls_pred, dim=1)[:, 1].cpu().numpy()
            auc = roc_auc_score(labels, probs)
        else:
            auc = 0.0
    except:
        auc = 0.0
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'score': score,
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': cm
    }
def plot_confusion_matrix(cm, epoch, phase, save_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stable', 'Unstable'],
                yticklabels=['Stable', 'Unstable'])
    plt.title(f'Confusion Matrix - Epoch {epoch+1} ({phase})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'confusion_matrix_{phase}_epoch_{epoch+1}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
def train_model(model, train_loader, val_loader, config):
    logger = get_logger()
    loss_weights = {'classification': 1.0}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    scaler = GradScaler('cuda')
    history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [],
        'specificity': [], 'f1': [], 'score': [], 'auc': []
    }
    best_score = -float('inf')
    patience_counter = 0
    logger.info("="*80)
    logger.info("YOLO-CONVNEXTV2 TRAINING STARTED")
    logger.info("="*80)
    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (c_img, w_img, stability_labels) in enumerate(train_bar):
            c_img = c_img.to(config.device)
            w_img = w_img.to(config.device)
            stability_labels = stability_labels.to(config.device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(c_img, w_img)
                losses, total_loss = model.compute_losses(outputs, stability_labels, loss_weights)
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += losses['total']
                train_preds.append(outputs['cls_pred_f'].detach())
                train_labels.append(stability_labels)
            else:
                logger.warning(f"NaN/Inf detected in batch {batch_idx}, skipping")
            train_bar.set_postfix({'loss': f"{losses['total']:.4f}"})
        train_loss /= len(train_loader)
        if train_preds:
            train_preds = torch.cat(train_preds)
            train_labels = torch.cat(train_labels)
            train_metrics = compute_metrics(train_preds, train_labels)
        else:
            train_metrics = compute_metrics(None, None)
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for c_img, w_img, stability_labels in val_bar:
                c_img = c_img.to(config.device)
                w_img = w_img.to(config.device)
                stability_labels = stability_labels.to(config.device)
                with autocast('cuda'):
                    outputs = model(c_img, w_img)
                    losses, total_loss = model.compute_losses(outputs, stability_labels, loss_weights)
                val_loss += losses['total']
                val_preds.append(outputs['cls_pred_f'])
                val_labels.append(stability_labels)
                val_bar.set_postfix({'loss': f"{losses['total']:.4f}"})
        val_loss /= len(val_loader)
        if val_preds:
            val_preds = torch.cat(val_preds)
            val_labels = torch.cat(val_labels)
            val_metrics = compute_metrics(val_preds, val_labels)
        else:
            val_metrics = compute_metrics(None, None)
        history['loss'].append(val_loss)
        for key in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'score', 'auc']:
            history[key].append(val_metrics[key])
        scheduler.step()
        logger.info("Training Results:")
        logger.info(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        logger.info("  Classification Metrics (Validation):")
        logger.info(f"    Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"    Precision: {val_metrics['precision']:.4f}")
        logger.info(f"    Recall (Sensitivity): {val_metrics['recall']:.4f}")
        logger.info(f"    Specificity: {val_metrics['specificity']:.4f}")
        logger.info(f"    F1-Score: {val_metrics['f1']:.4f}")
        logger.info(f"    AUC: {val_metrics['auc']:.4f}")
        logger.info(f"    Composite Score (P×S×R): {val_metrics['score']:.4f}")
        current_score = val_metrics['score']
        if current_score > best_score or epoch == 0:
            best_score = current_score
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config
            }, config.get_model_save_path())
            logger.info(f"  *** NEW BEST MODEL SAVED ***")
            logger.info(f"  Best Score: {best_score:.6f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement for {patience_counter} epochs")
        if (epoch + 1) % 10 == 0:
            try:
                plot_confusion_matrix(train_metrics['confusion_matrix'], epoch, 'train', config.save_dir)
                plot_confusion_matrix(val_metrics['confusion_matrix'], epoch, 'val', config.save_dir)
                logger.info(f"  Confusion matrices saved for epoch {epoch+1}")
            except Exception as e:
                logger.warning(f"Failed to save confusion matrix: {str(e)}")
        if patience_counter >= config.patience:
            logger.info(f"  Early stopping triggered after {config.patience} epochs without improvement")
            break
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Total Epochs Completed: {epoch + 1}")
    logger.info(f"Best Score Achieved: {best_score:.6f}")
    logger.info("="*80)
    return history, best_score
def evaluate_model(model, val_loader, config):
    logger = get_logger()
    checkpoint = torch.load(config.get_model_save_path(), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("\n" + "="*80)
    logger.info("FINAL MODEL EVALUATION")
    logger.info("="*80)
    logger.info(f"Loaded best model with score: {checkpoint['best_score']:.6f}")
    all_preds = []
    all_labels = []
    all_probs = []
    logger.info("Performing final evaluation on validation set...")
    with torch.no_grad():
        for c_img, w_img, stability_labels in tqdm(val_loader, desc="Final evaluation"):
            c_img = c_img.to(config.device)
            w_img = w_img.to(config.device)
            stability_labels = stability_labels.to(config.device)
            outputs = model(c_img, w_img)
            cls_logits = outputs['cls_pred_f']
            probs = F.softmax(cls_logits, dim=1)
            all_preds.append(cls_logits.cpu())
            all_labels.extend(stability_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_labels_tensor = torch.tensor(all_labels)
    final_metrics = compute_metrics(all_preds_tensor, all_labels_tensor)
    logger.info("="*80)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"Dataset Size: {len(all_labels)} samples")
    logger.info(f"Model Performance Metrics:")
    logger.info(f"  Accuracy:    {final_metrics['accuracy']:.4f}")
    logger.info(f"  Precision:   {final_metrics['precision']:.4f}")
    logger.info(f"  Recall:      {final_metrics['recall']:.4f}")
    logger.info(f"  Specificity: {final_metrics['specificity']:.4f}")
    logger.info(f"  F1-Score:    {final_metrics['f1']:.4f}")
    logger.info(f"  AUC:         {final_metrics['auc']:.4f}")
    logger.info(f"  Composite Score (P×S×R): {final_metrics['score']:.4f}")
    logger.info("="*80)
    return final_metrics, all_preds, all_labels, all_probs
def main():
    logger = setup_logging()
    logger.info("="*100)
    logger.info("YOLO-CONVNEXTV2 VITILIGO CLASSIFICATION MODEL TRAINING")
    logger.info("="*100)
    if not os.path.exists(Config.clinic_yolo_path):
        logger.error(f"Clinic YOLO model not found: {Config.clinic_yolo_path}")
        return
    if not os.path.exists(Config.wood_yolo_path):
        logger.error(f"Wood YOLO model not found: {Config.wood_yolo_path}")
        return
    logger.info("Initializing YOLO ROI extractor...")
    roi_extractor = YOLOROIExtractor(Config.clinic_yolo_path, Config.wood_yolo_path, Config.device)
    logger.info("Initializing dataset...")
    base_dataset = YOLOVitiligoDataset(
        root_dir=Config.root_dir,
        csv_file=Config.csv_file,
        roi_extractor=roi_extractor,
        image_size=(Config.roi_size, Config.roi_size)
    )
    logger.info(f"Original dataset size: {len(base_dataset)} pairs")
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    train_indices, val_indices = train_test_split(
        range(len(base_dataset)),
        test_size=val_size,
        random_state=42,
        stratify=[base_dataset.group_keys[i][1] for i in range(len(base_dataset))]
    )
    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(base_dataset, val_indices)
    logger.info(f"Training set size: {len(train_dataset)} pairs")
    logger.info(f"Validation set size: {len(val_dataset)} pairs")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    logger.info(f"Training data loader: {len(train_loader)} batches")
    logger.info(f"Validation data loader: {len(val_loader)} batches")
    logger.info("Creating YOLO-ConvNeXtV2 model...")
    model = YOLOConvNeXtV2Model(Config).to(Config.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model created successfully!")
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    history, best_score = train_model(model, train_loader, val_loader, Config)
    final_metrics, predictions, true_labels, probabilities = evaluate_model(model, val_loader, Config)
    logger.info("YOLO-CONVNEXTV2 TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*100)
if __name__ == "__main__":
    main()

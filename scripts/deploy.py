import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from ultralytics import YOLO 
import torch.nn.functional as F
import timm


class VMTLConfig:
    img_size = 224
    hidden_dim = 384
    num_classes = 2
    dropout_rate = 0.5
    model_path = "../outputs/checkpoints/proposed/V6/best_model_P0.910_S0.934_R0.909.pth"

class ConvQKVAttentionVMTL(nn.Module):
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
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, config.img_size, config.img_size)
            self.feature_dims = [f.shape[1] for f in self.backbone(dummy)]
            
        self.cross_attention = ConvQKVAttentionVMTL(
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
            nn.Conv2d(256, 4, 1)  # 4 channels for bbox coordinates
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

    def forward(self, clinic_img, wood_img):
        clinic_feat = self.backbone(clinic_img)[-1]
        wood_feat = self.backbone(wood_img)[-1]
        fused_features, (clinic_attn, wood_attn), (z_c, z_w, z_f) = self.cross_attention(clinic_feat, wood_feat)
        fused = self.fusion_conv(fused_features)

        return {
            'classification': self.classifier(fused),
            'detection': self.detection_head(fused),
            'segmentation': self.segmentation_head(fused),
            'reconstruction': self.reconstruction_head(fused)
        }


class Config:
    img_size = 224
    roi_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    clinic_yolo_path = "../outputs/checkpoints/proposed/best_det.pt"
    wood_yolo_path = "../outputs/checkpoints/proposed/best_seg.pt"
    model_weights_path = "../outputs/checkpoints/proposed/best_yolo_convnext_model.pth"
    
    encoder_name = "convnextv2_tiny"
    hidden_dim = 384
    num_classes = 2
    dropout_rate = 0.5

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
       
        self.cross_attention = ConvQKVAttentionVMTL(
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


def load_image_from_input(file_obj):
    if file_obj is None:
        return None
    
    if isinstance(file_obj, Image.Image):
        image = np.array(file_obj)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image
    elif hasattr(file_obj, 'name'):
        image = cv2.imread(file_obj.name)
        if image is None:
            raise ValueError("Could not load image.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    else:
        raise ValueError("Unsupported input type")

def image_to_pil(image_array):
    if image_array is None:
        return None
    return Image.fromarray(image_array.astype('uint8'), 'RGB')

def preprocess_for_model(image, target_size=(224, 224)):
    if image is None:
        return torch.zeros(1, 3, *target_size, dtype=torch.float32)

    image_resized = cv2.resize(image, target_size)
    image_tensor = torch.tensor(image_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    return image_tensor.unsqueeze(0)

class VitiligoApp:
    def __init__(self):
        self.config = Config()
        self.device = self.config.device
        print(f"Using device: {self.device}")


        print("Loading YOLO models for VMSL...")
        if not os.path.exists(self.config.clinic_yolo_path):
            raise FileNotFoundError(f"Clinic YOLO model not found at {self.config.clinic_yolo_path}")
        if not os.path.exists(self.config.wood_yolo_path):
            raise FileNotFoundError(f"Wood YOLO model not found at {self.config.wood_yolo_path}")
            
        self.clinic_yolo = YOLO(self.config.clinic_yolo_path)
        self.wood_yolo = YOLO(self.config.wood_yolo_path)
        print("YOLO models loaded for VMSL.")


        print("Loading classifier model for VMSL...")
        if not os.path.exists(self.config.model_weights_path):
            raise FileNotFoundError(f"VMSL classifier model weights not found at {self.config.model_weights_path}")
            
        self.classifier_model = YOLOConvNeXtV2Model(self.config).to(self.device)
        
        try:
            checkpoint = torch.load(self.config.model_weights_path, map_location=self.device, weights_only=False)
            self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"VMSL classifier model weights loaded (Epoch: {checkpoint.get('epoch', 'N/A')}, Best Score: {checkpoint.get('best_score', 'N/A'):.4f}).")
        except Exception as e:
            print(f"Error loading VMSL classifier model weights: {e}")
            raise
            
        self.classifier_model.eval()
        print("VMSL classifier model is ready.")
        

        print("Loading VMTL model...")
        vmtl_config = VMTLConfig()
        self.vmtl_model = MultiTaskModel(vmtl_config).to(self.device)
        
        try:
            checkpoint = torch.load(vmtl_config.model_path, map_location=self.device, weights_only=False)


            model_state_dict = self.vmtl_model.state_dict()
            filtered_checkpoint = {
                k: v for k, v in checkpoint['model_state_dict'].items() 
                if k in model_state_dict
            }


            self.vmtl_model.load_state_dict(filtered_checkpoint, strict=False)
            print(f"VMTL model loaded successfully from {vmtl_config.model_path}.")
        except Exception as e:
            print(f"Error loading VMTL model weights: {e}")
            raise
            
        self.vmtl_model.eval()
        print("VMTL model is ready.")

    def process_images_vmsl(self, clinic_file, wood_file):

        try:
            report_md = ""  
            
            clinic_img_orig = load_image_from_input(clinic_file)
            wood_img_orig = load_image_from_input(wood_file)

            if clinic_img_orig is None or wood_img_orig is None:
                raise ValueError("Failed to load one or both images.")


            save_dir = os.path.join("..","outputs","results","runs", "detect", "predict")
            if os.path.exists(save_dir):
                for file in os.listdir(save_dir):
                    if file.endswith('.jpg') or file.endswith('.png'):
                        os.remove(os.path.join(save_dir, file))


            clinic_results = self.clinic_yolo.predict(
                source=clinic_img_orig, 
                conf=0.25, 
                iou=0.45, 
                save=True,
                project="../outputs/results/runs",
                name="detect/predict",
                exist_ok=True
            )
            

            clinic_img_display = clinic_img_orig.copy()
            if len(clinic_results) > 0 and hasattr(clinic_results[0], 'plot'):
                clinic_img_display = clinic_results[0].plot()
                clinic_img_display = cv2.cvtColor(clinic_img_display, cv2.COLOR_BGR2RGB)
            

            clinic_roi = clinic_img_orig
            clinic_confidence = 0.0
            if len(clinic_results) > 0 and hasattr(clinic_results[0], 'boxes') and clinic_results[0].boxes is not None and len(clinic_results[0].boxes) > 0:
                boxes = clinic_results[0].boxes
                if len(boxes) > 0:
                    box = boxes.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    if (y2 - y1) > 10 and (x2 - x1) > 10:
                        clinic_roi = clinic_img_orig[y1:y2, x1:x2]
                    clinic_confidence = boxes.conf[0].item() if hasattr(boxes, 'conf') and len(boxes.conf) > 0 else 0.0


            save_dir = os.path.join("..","outputs","results","runs", "segment", "predict")
            if os.path.exists(save_dir):
                for file in os.listdir(save_dir):
                    if file.endswith('.jpg') or file.endswith('.png'):
                        os.remove(os.path.join(save_dir, file))


            wood_results = self.wood_yolo.predict(
                source=wood_img_orig,
                conf=0.25,
                iou=0.45,
                save=True,
                project="../outputs/results/runs",
                name="segment/predict",
                exist_ok=True
            )
            

            wood_img_display = wood_img_orig.copy()
            mask_found = False
            if len(wood_results) > 0 and hasattr(wood_results[0], 'plot'):
                wood_img_display = wood_results[0].plot()
                wood_img_display = cv2.cvtColor(wood_img_display, cv2.COLOR_BGR2RGB)
                mask_found = True
            

            wood_roi = wood_img_orig
            if mask_found and hasattr(wood_results[0], 'masks') and wood_results[0].masks is not None:
                masks = wood_results[0].masks.data.cpu().numpy()
                if len(masks) > 0:
                    mask = masks[0]
                    coords = np.where(mask > 0.5)
                    if len(coords[0]) > 0:
                        y_min, y_max = coords[0].min(), coords[0].max()
                        x_min, x_max = coords[1].min(), coords[1].max()
                        roi = wood_img_orig[y_min:y_max, x_min:x_max]
                        if roi.shape[0] > 10 and roi.shape[1] > 10:
                            wood_roi = roi
            elif mask_found:
                report_md += "\n⚠️ Wood lamp image segmentation did not find any specific regions. Using full image for analysis."


            clinic_tensor = preprocess_for_model(clinic_roi, (self.config.roi_size, self.config.roi_size)).to(self.device)
            wood_tensor = preprocess_for_model(wood_roi, (self.config.roi_size, self.config.roi_size)).to(self.device)


            with torch.no_grad():
                outputs = self.classifier_model(clinic_tensor, wood_tensor)
                logits = outputs['cls_pred_f']
                probabilities = F.softmax(logits, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                predicted_probability = probabilities[0][predicted_class_idx].item()


            class_names = {0: "Stable", 1: "Non-Stable"}
            predicted_class_name = class_names[predicted_class_idx]

            report_md = f"""
            ## 📋 VMSL Diagnosis Report

            ## 🩺 Classification Result
            - **Prediction:** **{predicted_class_name}**
            - **Confidence:** {predicted_probability*100:.2f}%

            
            """

        except Exception as e:
            error_msg = f"An error occurred during VMSL processing: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, None, error_msg

        return image_to_pil(clinic_img_display), image_to_pil(wood_img_display), report_md

    def process_images_vmtl(self, clinic_file, wood_file):
        try:
            report_md = ""
    
            clinic_img_orig = load_image_from_input(clinic_file)
            wood_img_orig = load_image_from_input(wood_file)
    
            if clinic_img_orig is None or wood_img_orig is None:
                raise ValueError("Failed to load one or both images.")
    
            orig_clinic_h, orig_clinic_w = clinic_img_orig.shape[:2]
            orig_wood_h, orig_wood_w = wood_img_orig.shape[:2]
    
            clinic_img = cv2.resize(clinic_img_orig, (224, 224))
            wood_img = cv2.resize(wood_img_orig, (224, 224))
    
            clinic_tensor = preprocess_for_model(clinic_img, (224, 224)).to(self.device)
            wood_tensor = preprocess_for_model(wood_img, (224, 224)).to(self.device)

            with torch.no_grad():
                outputs = self.vmtl_model(clinic_tensor, wood_tensor)
        
                pred_class = F.softmax(outputs['classification'], dim=1)
                class_probs = pred_class[0].cpu().numpy()
                pred_label_idx = torch.argmax(pred_class, dim=1).item()
                pred_label = "Stable" if pred_label_idx == 0 else "non-Stable"

                detection_output = outputs['detection']
                pred_bboxes = self.process_detection_output(detection_output)[0]
        
                segmentation_output = torch.sigmoid(outputs['segmentation'])
                pred_mask_np = segmentation_output[0, 0].cpu().numpy()
        
                reconstruction_output = outputs['reconstruction']
    

            clinic_detection = clinic_img_orig.copy()
            if pred_bboxes:
                clinic_detection = self.draw_bboxes_on_image(clinic_detection, pred_bboxes, color=(255, 0, 0))
            clinic_detection_pil = image_to_pil(clinic_detection)


            pred_mask_resized = cv2.resize(
                pred_mask_np, 
                (orig_wood_w, orig_wood_h),
                interpolation=cv2.INTER_LINEAR
            )
    
            pred_colored_mask = np.zeros((orig_wood_h, orig_wood_w, 3), dtype=np.uint8)
            pred_colored_mask[pred_mask_resized > 0.5] = [255, 0, 0]
    
            alpha = 0.3
            wood_segmentation = cv2.addWeighted(
                wood_img_orig, 
                1 - alpha, 
                pred_colored_mask, 
                alpha, 
                0
            )
            wood_segmentation_pil = image_to_pil(wood_segmentation)


            reconstruction_img = reconstruction_output[0].permute(1, 2, 0).cpu().numpy()
            reconstruction_img = (reconstruction_img * 255).clip(0, 255).astype(np.uint8)
            clinic_reconstruction_pil = Image.fromarray(reconstruction_img)
    
            report_md = f"""
            ## 📋 VMTL Diagnosis Report

            ### 🩺 Classification Result
            - **Prediction:** **{pred_label}**
            - **Stable Probability:** {class_probs[0]*100:.2f}%
            - **Progressive Probability:** {class_probs[1]*100:.2f}%

            """
    
            return clinic_reconstruction_pil, clinic_detection_pil, wood_segmentation_pil, report_md
    
        except Exception as e:
            error_msg = f"An error occurred during VMTL processing: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, None, None, error_msg


    def process_detection_output(self, detection_output, threshold=0.5):
        batch_size, _, H, W = detection_output.shape
        bboxes_batch = []
    
        for b in range(batch_size):
            pred = detection_output[b]
        

            xc = pred[0].mean().item()
            yc = pred[1].mean().item()
            w = pred[2].mean().item()
            h = pred[3].mean().item()
        

            img_size = 224
            x1 = max(0, (xc - w/2) * img_size)
            y1 = max(0, (yc - h/2) * img_size)
            x2 = min(img_size, (xc + w/2) * img_size)
            y2 = min(img_size, (yc + h/2) * img_size)
        
            bbox = []
            if w > 0.1 and h > 0.1:
                bbox = [[x1, y1, x2, y2]]
        

            bboxes_batch.append(bbox)
    
        return bboxes_batch

    def draw_bboxes_on_image(self, image, bboxes, color=(255, 0, 0), thickness=2):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        return np.array(image_with_boxes)

def create_interface():

    app = VitiligoApp()
    
    with gr.Blocks(
        title="Vitiligo Diagnosis System",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="pink")
    ) as demo:
        gr.Markdown("# 🩺 Vitiligo Diagnosis System")
        
        gr.Markdown("### 📋 Introduction to Framework Diagram")

        with gr.Tab("VMSL"):
            with gr.Row():
                gr.Image(
                    value="../datasets/raw/framework.png", 
                    label="🖼️ frame diagram",
                    interactive=False, 
                    height=300
                )


            gr.Markdown("### 📋 The Vitiligo Multiple-stage Learning framework comprises three core stages (S1-S3), utilizing three DMT2 modules, one BINDA1 module, and one ITC03 module, spanning an extended workflow from stage S0 to S3, and implementing both Finetune and Frozen training strategies.")


            with gr.Row():
                clinic_input = gr.Image(label="📷 Clinic Image", type="pil")
                wood_input = gr.Image(label="🔦 Wood Lamp Image", type="pil")

            with gr.Row():
                vmsl_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

            gr.Markdown("### 🖼️ Detection and Segmentation Results")
            with gr.Row():
                clinic_output = gr.Image(label="Clinic Image", interactive=False, height=400)
                wood_output = gr.Image(label="Wood Image", interactive=False, height=400)
        

            gr.Markdown("### 📋 Diagnosis Report")
            report_output = gr.Markdown(label="Report", value="Upload images and click 'Analyze'.")

 
            vmsl_btn.click(
                fn=app.process_images_vmsl,
                inputs=[clinic_input, wood_input],
                outputs=[clinic_output, wood_output, report_output]
            )

        with gr.Tab("VMTL"):
            with gr.Row():
                gr.Image(
                    value="../datasets/raw/VMTL.png", 
                    label="🖼️ frame diagram",
                    interactive=False, 
                    height=300
                )


            gr.Markdown("### 📋 The Vitiligo Multiple-tasking Learning (VMTL) framework uses shared weights across an S0-S3 backbone with auxiliary modules (DMPP, F4-F6), implementing phase-driven learning (constriction, reconstruction, decision) to support three tasks: segmentation, stable/non-stable analysis, and classification, trained via Finetune/Frozen strategies.")


            with gr.Row():
                clinic_input = gr.Image(label="📷 Clinic Image", type="pil")
                wood_input = gr.Image(label="🔦 Wood Lamp Image", type="pil")

            with gr.Row():
                vmsl_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

            gr.Markdown("### 🖼️ Detection and Segmentation Results")

            with gr.Row():
                reconstruction_output = gr.Image(label="Clinic Reconstruction", interactive=False, height=300)
                detection_output = gr.Image(label="Clinic Detection", interactive=False, height=300)
                segmentation_output = gr.Image(label="Wood Segmentation", interactive=False, height=300)


            gr.Markdown("### 📋 Diagnosis Report")
            report_output = gr.Markdown(label="Report", value="Upload images and click 'Analyze'.")


            vmsl_btn.click(
                fn=app.process_images_vmtl,
                inputs=[clinic_input, wood_input],
                outputs=[reconstruction_output, detection_output, segmentation_output, report_output]
            )

    

    return demo

if __name__ == "__main__":
    config = Config()
    vmtl_config = VMTLConfig()
    
    missing_files = []

    if not os.path.exists(config.clinic_yolo_path):
        missing_files.append(config.clinic_yolo_path)
    if not os.path.exists(config.wood_yolo_path):
        missing_files.append(config.wood_yolo_path)
    if not os.path.exists(config.model_weights_path):
        missing_files.append(config.model_weights_path)
    

    if not os.path.exists(vmtl_config.model_path):
        missing_files.append(vmtl_config.model_path)

    if missing_files:
        print("Error: The following required model files are missing:")
        for f in missing_files:
            print(f" - {f}")
        print("\nPlease ensure the files are in the correct location.")
        exit(1)

    print("Launching Gradio interface...")
    interface = create_interface()
    interface.launch(share=True)
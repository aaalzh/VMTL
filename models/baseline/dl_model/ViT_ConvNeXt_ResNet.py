#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt


# Vitiligo Dataset

# In[2]:


class VitiligoDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.grouped = self.data.groupby(['pair_id', 'stability'])

    def __len__(self):
        return len(self.grouped.groups)

    def __getitem__(self, idx):

        pair_id, stability = list(self.grouped.groups.keys())[idx]
        pairs = self.grouped.get_group((pair_id, stability))
        clinic = pairs[pairs['image_type'] == 'clinic']
        wood = pairs[pairs['image_type'] == 'wood']
        clinic_path =  clinic['image_path'].values[0]
        wood_path = wood['image_path'].values[0]
        clinic_image = Image.open(clinic_path).convert('RGB')
        wood_image = Image.open(wood_path).convert('RGB')
        if self.transform:
            clinic_image = self.transform(clinic_image)
            wood_image = self.transform(wood_image)
        label = 1 if stability == 'stable' else 0
        label = torch.tensor(label, dtype=torch.long)
        return clinic_image, wood_image, label



# Visulization

# In[3]:


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

dataset = VitiligoDataset(csv_file='../../../../datasets/data.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in dataloader:
    clinic_images, wood_images, labels = batch
    batch_size = len(labels)
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
    if batch_size == 1:
        axes = np.array([axes])
    for i in range(batch_size):
        clinic_img = clinic_images[i].permute(1, 2, 0).numpy()
        wood_img = wood_images[i].permute(1, 2, 0).numpy()
        label = 'Stable' if labels[i] == 1 else 'Non-Stable'
        axes[i, 0].imshow(clinic_img)
        axes[i, 0].set_title(f'Clinic Image - {label}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(wood_img)
        axes[i, 1].set_title(f'Wood Image - {label}')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()
    break



# Configs

# In[4]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model_name = 'resnet50'
batch_size = 32
num_epochs = 300
patience = 50
lr = 1e-3


# Split the data

# In[5]:


backbone = timm.create_model(model_name, pretrained=True)
transform = timm.data.create_transform(**timm.data.resolve_data_config(backbone.pretrained_cfg))
dataset = VitiligoDataset(csv_file='../../../../datasets/data.csv',  transform=transform)
train_ids, val_ids = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_ids)
val_dataset = Subset(dataset, val_ids)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Baseline

# In[6]:


class Baseline(nn.Module):
    def __init__(self, model_name='convnext_base', num_classes=2, dropout=0.5):
        super(Baseline, self).__init__()
        self.model_name = model_name.lower() 
        self.num_classes = num_classes

        if self.model_name == 'resnet50':
            self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.feature_dim = self.backbone.num_features
            self.backbone.fc = nn.Identity()

        elif self.model_name == 'vit_base_patch16_224':
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.feature_dim = self.backbone.num_features

        elif self.model_name == 'convnext_base':
            self.backbone = timm.create_model('convnext_base', pretrained=True, num_classes=0)
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.feature_dim = self.backbone.num_features
            self.backbone.head.fc = nn.Identity()  

        else:
            raise ValueError(f"Unsupported model name: {model_name}. Supported models are 'resnet50', 'vit_base_patch16_224', and 'convnext_base'.")

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * self.feature_dim, num_classes)

    def forward(self, clinic_image, wood_image):
        if self.model_name == 'resnet50':
            clinic_features = self.backbone(clinic_image)
            wood_features = self.backbone(wood_image)

        elif self.model_name == 'vit_base_patch16_224':
            clinic_features = self.backbone.forward_features(clinic_image)[:, 0, :]
            wood_features = self.backbone.forward_features(wood_image)[:, 0, :]

        elif self.model_name == 'convnext_base':
            clinic_features = self.backbone(clinic_image)
            wood_features = self.backbone(wood_image)

        combined_features = torch.cat((clinic_features, wood_features), dim=1)
        combined_features = self.dropout(combined_features)
        output = self.head(combined_features)
        return output


# Train and Validate

# In[7]:


def train_one_epoch(model, criterion, train_loader, optimizer, device=device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    for clinic_images, wood_images, labels in train_loader:
        clinic_images = clinic_images.to(device)
        wood_images = wood_images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(clinic_images, wood_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return running_loss / len(train_loader), precision, recall, auc, specificity, accuracy


def validate(model, val_loader, criterion, device=device):
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0  
    with torch.no_grad():
        for clinic_images, wood_images, labels in val_loader:
            clinic_images, wood_images, labels = clinic_images.to(device), wood_images.to(device), labels.to(device)
            outputs = model(clinic_images, wood_images)
            loss = criterion(outputs, labels)  
            running_loss += loss.item() 
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_loss = running_loss / len(val_loader)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return val_loss, precision, recall, auc, specificity, accuracy

def create_optimzer_with_weight_decay(model, lr=0.001,weight_decay=1e-5):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# Start Trainging

# In[8]:


baseline = Baseline(model_name=model_name).to(device)
for name, param in baseline.named_parameters():
    if param.requires_grad:
        print(f"total: {name}")


# In[9]:


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(baseline.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
writer = SummaryWriter('../../../../outputs/logs/baseline/model_base')
best_val_metrics = float('-inf')
counter = 0
best_model_dir = '../../../../outputs/checkpoints/baseline/saved_models'
os.makedirs(best_model_dir, exist_ok=True)

for epoch in tqdm(range(num_epochs), desc='Training Progress'):
    train_loss, train_precision, train_recall, train_auc, train_specificity, train_accuracy = train_one_epoch(
        baseline, criterion, train_loader, optimizer, device)
    val_loss, val_precision, val_recall, val_auc, val_specificity, val_accuracy = validate(baseline, val_loader, criterion, device)

    # 计算F1分数
    if val_precision + val_recall > 0:
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
    else:
        val_f1 = 0

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('val_loss', val_loss, epoch)
    writer.add_scalar('train_precision', train_precision, epoch)
    writer.add_scalar('val_precision', val_precision, epoch)
    writer.add_scalar('train_recall', train_recall, epoch)
    writer.add_scalar('val_recall', val_recall, epoch)
    writer.add_scalar('train_auc', train_auc, epoch)
    writer.add_scalar('val_auc', val_auc, epoch)
    writer.add_scalar('train_specificity', train_specificity, epoch)
    writer.add_scalar('val_specificity', val_specificity, epoch)
    writer.add_scalar('train_accuracy', train_accuracy, epoch)
    writer.add_scalar('val_accuracy', val_accuracy, epoch)
    writer.add_scalar('val_f1', val_f1, epoch)

    if epoch % 2 == 0:
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, '
              f'AUC: {train_auc:.4f}, Specificity: {train_specificity:.4f}, Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, '
              f'AUC: {val_auc:.4f}, Specificity: {val_specificity:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}')


    composite_metric = val_precision * val_specificity * val_recall
    scheduler.step(composite_metric)

    is_best = composite_metric > best_val_metrics
    if is_best:
        best_val_metrics = composite_metric
        best_val_precision = val_precision
        best_val_specificity = val_specificity
        best_val_f1 = val_f1
        best_val_auc = val_auc
        best_val_recall = val_recall
        best_val_accuracy = val_accuracy
        counter = 0
        model_path = os.path.join(best_model_dir,
                                  f"best_model_p{val_precision:.4f}_s{val_specificity:.4f}_f1{val_f1:.4f}.pth")
        torch.save(baseline.state_dict(), model_path)
        print(f"★ Saved best model: {os.path.basename(model_path)}")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

writer.close()     


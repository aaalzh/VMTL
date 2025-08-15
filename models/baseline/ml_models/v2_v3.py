#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#v3.0
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import cv2
from sklearn.metrics import confusion_matrix
from pycaret.classification import ClassificationExperiment
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings
import psutil
import gc
from sklearn.model_selection import StratifiedKFold

# 禁用警告
warnings.filterwarnings("ignore")

# 内存监控工具
def check_available_memory():
    mem = psutil.virtual_memory()
    return mem.available / (1024**3)  # GB

# --------------------------
# 数据加载与特征提取（保持不变）
# --------------------------
class PairedDataset:
    def __init__(self, csv_file, transform=None, return_pil=True):
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.grouped = self.data.groupby(['pair_id', 'stability'])
        self.return_pil = return_pil

    def __len__(self):
        return len(self.grouped.groups)

    def __getitem__(self, idx):
        pair_id, stability = list(self.grouped.groups.keys())[idx]
        pairs = self.grouped.get_group((pair_id, stability))
        clinic = pairs[pairs['image_type'] == 'clinic']
        wood = pairs[pairs['image_type'] == 'wood']

        try:
            clinic_path = clinic['image_path'].values[0]
            wood_path = wood['image_path'].values[0]
        except IndexError:
            print(f"Warning: Missing images for pair ID {pair_id}, skipping this sample")
            return None, None, None

        try:
            clinic_image = Image.open(clinic_path).convert('RGB')
            wood_image = Image.open(wood_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Cannot open image {clinic_path} or {wood_path}: {e}, skipping this sample")
            return None, None, None

        pil_clinic = clinic_image
        pil_wood = wood_image

        if self.transform:
            clinic_image = self.transform(clinic_image)
            wood_image = self.transform(wood_image)

        label = 1 if stability == 'stable' else 0
        label = torch.tensor(label, dtype=torch.long)

        if self.return_pil:
            return pil_clinic, pil_wood, label
        else:
            return clinic_image, wood_image, label


def extract_glcm_features(image):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    image = np.array(image.convert('L'))  # Convert to grayscale

    # GLCM feature parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Extract 6 GLCM features + entropy
    contrast = graycoprops(glcm, prop='contrast').mean()
    dissimilarity = graycoprops(glcm, prop='dissimilarity').mean()
    homogeneity = graycoprops(glcm, prop='homogeneity').mean()
    asm = graycoprops(glcm, prop='ASM').mean()
    energy = graycoprops(glcm, prop='energy').mean()
    correlation = graycoprops(glcm, prop='correlation').mean()
    entropy = shannon_entropy(image)

    return np.array([contrast, dissimilarity, homogeneity, asm, energy, correlation, entropy])


def extract_contour_features(image):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    image = np.array(image)
    original = image.copy()

    # 1. 转换到LAB颜色空间并提取A通道
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a_original, b = cv2.split(lab)  # 重命名为a_original便于区分

    # 2. 创建偏白色区域掩码 (L > 70 且 |a| < 10 且 |b| < 10)
    white_mask = np.logical_and.reduce([
        l > 70,                   # 亮度较高
        np.abs(a_original - 128) < 10,  # a通道接近中性
        np.abs(b - 128) < 10          # b通道接近中性
    ])

    # 3. 基础对比度增强
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    a_enhanced_base = clahe.apply(a_original)

    # 4. 识别从偏白色变为黑色的区域
    # 将a通道转换到-128~127范围便于理解颜色含义
    a_original_signed = a_original.astype(np.int16) - 128
    a_enhanced_signed = a_enhanced_base.astype(np.int16) - 128

    # 偏白色区域变为更绿色/红色（负值更大）的像素
    turn_green_mask = np.logical_and(white_mask, a_enhanced_signed < a_original_signed - 5)

    # 5. 对这些区域进一步增强对比度
    a_enhanced = a_enhanced_base.copy()

    # 创建区域特异性CLAHE，对目标区域应用更强的对比度增强
    target_pixels = a_original[turn_green_mask]
    if len(target_pixels) > 0:
        # 为目标区域创建专用CLAHE
        clahe_target = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
        a_target_enhanced = clahe_target.apply(a_original)

        # 仅替换目标区域
        a_enhanced[turn_green_mask] = a_target_enhanced[turn_green_mask]

    # 6. 动态范围拉伸（使黑色更黑）
    pixels = a_enhanced.flatten()
    p2, p98 = np.percentile(pixels, (2, 98))
    a_stretched = cv2.convertScaleAbs(a_enhanced, alpha=255/(p98-p2), beta=-255*p2/(p98-p2))

    # 7. 创建像素筛选掩码（仅保留连续变黑的区域）
    # 条件：a_enhanced < a_original 且 a_stretched < a_enhanced 且 a_stretched < a_original
    mask_1 = a_enhanced < a_original  # a_enhanced比original黑
    mask_2 = a_stretched < a_enhanced  # a_stretched比a_enhanced黑
    mask_3 = a_stretched < a_original  # a_stretched比original黑

    # 合并三个掩码
    final_mask = np.logical_and(np.logical_and(mask_1, mask_2), mask_3)
    final_mask = np.uint8(final_mask) * 255  # 转换为0-255的掩码

    # 8. 形态学优化掩码
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 9. 查找轮廓并绘制
    contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 提取轮廓特征
    min_contour_area = 100  # Minimum contour area threshold
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    sorted_contours = sorted(filtered_contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)  # Sort by area

    # Keep only top 50 contours, pad with zeros if insufficient
    features = []
    for cnt in sorted_contours[:50]:  # Take top 50 contours
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        features.extend([area, perimeter])

    # Calculate padding needed
    total_needed = 50 * 2  # Fixed 100 features (50 contours x 2 features)
    missing = total_needed - len(features)
    if missing > 0:
        features += [0.0] * missing  # Pad with zeros

    return np.array(features)


def get_features(dataset):
    print(f"Starting feature extraction, available memory: {check_available_memory():.2f} GB")

    features = []
    labels = []

    feature_dataset = PairedDataset(
        csv_file=dataset.csv_file,  
        transform=None, 
        return_pil=True
    )

    for idx in tqdm(range(len(feature_dataset)), desc="Extracting features"):
        if idx % 100 == 0:
            mem = check_available_memory()
            print(f"Processed {idx}/{len(feature_dataset)} samples, available memory: {mem:.2f} GB")
            if mem < 1.0:
                print("Warning: Low system memory, potential crash risk!")

        clinic_image, wood_image, label = feature_dataset[idx]

        if wood_image is None or clinic_image is None:
            continue

        # Extract features
        clinic_glcm = extract_glcm_features(clinic_image)
        wood_glcm = extract_glcm_features(wood_image)
        clinic_contour = extract_contour_features(clinic_image)
        wood_contour = extract_contour_features(wood_image)

        # Concatenate features
        concatenated_features = np.concatenate((
            clinic_glcm, clinic_contour,
            wood_glcm, wood_contour
        ))
        features.append(concatenated_features)
        labels.append(label.item())

        if idx % 500 == 0:
            gc.collect()
            print(f"Memory cleaned, available memory: {check_available_memory():.2f} GB")

    print(f"Feature extraction completed, available memory: {check_available_memory():.2f} GB")
    return np.array(features), np.array(labels)

# Calculate specificity
def calculate_specificity(y_true, y_pred):
    """Calculate specificity = TN / (TN + FP)"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size < 4:  # Ensure binary classification
        return 0.0
    tn, fp, _, _ = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# 优化的特征选择函数
def select_features(X, y, n_features=100):
    """使用方差分析(ANOVA)选择最相关的特征"""
    from sklearn.feature_selection import SelectKBest, f_classif

    print(f"Before feature selection: {X.shape[1]} features")
    selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    print(f"After feature selection: {X_selected.shape[1]} features")

    return X_selected, selector

# 主函数
def main():
    # --------------------------
    # 参数配置
    # --------------------------
    params = {
        'val_size': 0.2,          # 验证集比例
        'n_iter_search': 20,      # 增加超参数搜索迭代次数
        'cv_folds': 5,            # 交叉验证折数
        'n_jobs': 1,              # 禁用并行计算
        'sample_frac': 1.0,       # 使用全部数据
        'n_features': 150,        # 特征选择保留的特征数
        'use_feature_selection': True  # 是否使用特征选择
    }

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])

    # 内存监控
    print(f"Starting execution, total memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Available memory: {check_available_memory():.2f} GB")

    # --------------------------
    # 特征提取（执行一次）
    # --------------------------
    print("===== Starting feature extraction (executed once) =====")
    csv_file = 'data.csv'
    dataset = PairedDataset(csv_file, transform=transform, return_pil=False)
    X, y = get_features(dataset)

    # 特征选择（优化）
    if params['use_feature_selection']:
        X, selector = select_features(X, y, n_features=params['n_features'])

    # 创建特征DataFrame
    features_df = pd.DataFrame(X)
    features_df['label'] = y

    # 内存监控
    print(f"Feature data size: {features_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"Feature extraction completed, available memory: {check_available_memory():.2f} GB")

    # 释放内存
    del dataset
    gc.collect()

    # --------------------------
    # 获取所有可用模型
    # --------------------------
    try:
        # 尝试使用最新版API
        exp = ClassificationExperiment()
        exp.setup(
            data=features_df.sample(frac=0.01, random_state=42),
            target='label',
            verbose=False,
            session_id=42
        )
        all_models = exp.models()
        model_ids = all_models['ID'].dropna().tolist()
        exp = ClassificationExperiment()  # 重置实验
        print(f"Successfully retrieved {len(model_ids)} models using latest PyCaret API")
    except Exception as e:
        print(f"Failed to get models via API: {e}. Using predefined list.")
        model_ids = ['lr', 'svm', 'rf', 'xgboost', 'lightgbm', 'dt', 'knn', 'nb', 'et', 'gbc', 'catboost']

    # 确保安装了catboost
    if 'catboost' in model_ids:
        try:
            import catboost
            print("CatBoost is installed and will be included in training.")
        except ImportError:
            print("CatBoost not installed. Skipping catboost.")
            model_ids.remove('catboost')

    print(f"\n===== Found {len(model_ids)} models, training once each =====")
    print(f"Models: {', '.join(model_ids)}")

    if not model_ids:
        print("No models available. Exiting.")
        return

    # --------------------------
    # 定义优化后的超参数网格
    # --------------------------
    optimized_param_grids = {
        'lr': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2', 'elasticnet'],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [1000, 2000]
        },
        'svm': {  
                'C': [0.01, 0.1, 1, 10, 100],  
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],  
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  
                'degree': [2, 3, 4] 
        },
        'rf': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'xgboost': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        },
        'lightgbm': {
            'n_estimators': [50, 100, 200],  
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [-1, 3, 5, 10], 
            'num_leaves': [15, 31, 50, 100],  
            'min_child_samples': [1, 3, 5, 10],  
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],  
            'reg_beta': [0, 0.1, 0.5, 1.0],  
            'min_split_gain': [0, 0.01, 0.1]  
                },
        'catboost': {
            'iterations': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 10],
            'border_count': [32, 64, 128]
        },
        'dt': {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1=曼哈顿距离，2=欧氏距离
        },
        'nb': {
            'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
        },
        'et': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'gbc': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
    }

    # --------------------------
    # 训练所有模型（各一次）
    # --------------------------
    all_results = {}
    failed_models = []

    for model_name in model_ids:
        print(f"\n{'='*70}")
        print(f"===== Training {model_name.upper()} (once with optimized parameters) =====")
        print(f"{'='*70}\n")

        try:
            seed = 42
            print(f"Training with seed: {seed} - {model_name.upper()}")
            print(f"Current available memory: {check_available_memory():.2f} GB")

            # 使用全部数据
            sample_df = features_df.sample(frac=params['sample_frac'], random_state=seed)

            # 初始化实验
            exp = ClassificationExperiment()

            # 设置实验
            exp.setup(
                data=sample_df,
                target='label',
                train_size=1-params['val_size'],
                normalize=True,
                normalize_method='zscore',
                fix_imbalance=True,
                fold=params['cv_folds'],
                session_id=seed,
                verbose=False,
                n_jobs=params['n_jobs']
            )

            # 添加特异性指标
            exp.add_metric(
                id='specificity',
                name='Specificity',
                score_func=calculate_specificity,
                greater_is_better=True
            )

            # 创建基础模型
            model = exp.create_model(model_name, verbose=False)

            # 超参数调优（使用优化后的网格）
            grid = optimized_param_grids.get(model_name, None)

            try:
                # 使用优化的随机搜索
                tuned_model = exp.tune_model(
                    estimator=model,
                    search_algorithm='random',
                    n_iter=params['n_iter_search'],
                    optimize='AUC',  # 优化AUC指标
                    custom_grid=grid,
                    verbose=False
                )
                print(f"Hyperparameter tuning completed for {model_name.upper()}")
            except Exception as e:
                print(f"Hyperparameter tuning failed: {e}. Using base model.")
                tuned_model = model

            # 评估模型
            exp.predict_model(tuned_model)
            metrics = exp.pull()

            # 提取关键指标（修正Precision为Prec.）
            required_metrics = ['AUC', 'Prec.', 'Recall', 'Specificity', 'Accuracy', 'F1']
            model_metrics = {}

            for metric in required_metrics:
                if metric in metrics.columns:
                    model_metrics[metric] = round(metrics[metric].iloc[0], 4)
                else:
                    print(f"Warning: Metric '{metric}' not found for {model_name}")
                    model_metrics[metric] = None

            # 保存结果
            all_results[model_name] = model_metrics
            print(f"{model_name.upper()} metrics: {model_metrics}")

            # 清理内存
            del model, tuned_model, exp, sample_df
            gc.collect()

        except Exception as e:
            print(f"Failed to train {model_name}: {str(e)[:100]}...")
            failed_models.append(model_name)
            continue

    # --------------------------
    # 整合结果为一个表格
    # --------------------------
    print(f"\n{'='*100}")
    print(f"===== All Models Training Results =====")
    print(f"Successfully trained: {len(all_results)}/{len(model_ids)} models")
    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")
    print(f"{'='*100}\n")

    # 创建结果表格
    if all_results:
        # 转换为DataFrame
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Model'

        # 按AUC排序
        results_df = results_df.sort_values(by='AUC', ascending=False)

        print("===== Model Performance Comparison (sorted by AUC) =====")
        print(results_df)

        # 保存表格为CSV
        results_df.to_csv('optimized_model_performance.csv')
        print("\nResults saved to 'optimized_model_performance.csv'")

    # --------------------------
    # 可视化：模型比较条形图
    # --------------------------
    if all_results and len(all_results) >= 2:
        plt.figure(figsize=(12, 8))
        results_df = pd.DataFrame.from_dict(all_results, orient='index')

        # 绘制主要指标条形图
        plot_metrics = ['AUC', 'Accuracy', 'F1', 'Specificity', 'Prec.', 'Recall']
        plot_data = results_df[plot_metrics].dropna(how='all')

        if not plot_data.empty:
            # 设置中文字体（避免警告）
            plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]

            ax = plot_data.plot(kind='bar', figsize=(15, 8))
            plt.title('模型性能比较（优化后）')
            plt.ylabel('得分')
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45)

            # 添加数值标签
            for p in ax.patches:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()
                ax.annotate(f'{height:.4f}', (x + width/2, y + height), ha='center', va='bottom', rotation=90)

            plt.tight_layout()
            plt.savefig('optimized_model_performance_comparison.png')
            print("Comparison plot saved to 'optimized_model_performance_comparison.png'")
            plt.close()

    print("\nAll tasks completed!")


if __name__ == "__main__":
    main()


# In[1]:


#v2.0
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import cv2
from sklearn.metrics import confusion_matrix
from pycaret.classification import ClassificationExperiment
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings
import psutil
import gc
from sklearn.model_selection import StratifiedKFold

# 禁用警告
warnings.filterwarnings("ignore")

# 内存监控工具
def check_available_memory():
    mem = psutil.virtual_memory()
    return mem.available / (1024**3)  # GB

# --------------------------
# 数据加载与特征提取（保持不变）
# --------------------------
class PairedDataset:
    def __init__(self, csv_file, transform=None, return_pil=True):
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.grouped = self.data.groupby(['pair_id', 'stability'])
        self.return_pil = return_pil

    def __len__(self):
        return len(self.grouped.groups)

    def __getitem__(self, idx):
        pair_id, stability = list(self.grouped.groups.keys())[idx]
        pairs = self.grouped.get_group((pair_id, stability))
        clinic = pairs[pairs['image_type'] == 'clinic']
        wood = pairs[pairs['image_type'] == 'wood']

        try:
            clinic_path = clinic['image_path'].values[0]
            wood_path = wood['image_path'].values[0]
        except IndexError:
            print(f"Warning: Missing images for pair ID {pair_id}, skipping this sample")
            return None, None, None

        try:
            clinic_image = Image.open(clinic_path).convert('RGB')
            wood_image = Image.open(wood_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Cannot open image {clinic_path} or {wood_path}: {e}, skipping this sample")
            return None, None, None

        pil_clinic = clinic_image
        pil_wood = wood_image

        if self.transform:
            clinic_image = self.transform(clinic_image)
            wood_image = self.transform(wood_image)

        label = 1 if stability == 'stable' else 0
        label = torch.tensor(label, dtype=torch.long)

        if self.return_pil:
            return pil_clinic, pil_wood, label
        else:
            return clinic_image, wood_image, label


def extract_glcm_features(image):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    image = np.array(image.convert('L'))  # Convert to grayscale

    # GLCM feature parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Extract 6 GLCM features + entropy
    contrast = graycoprops(glcm, prop='contrast').mean()
    dissimilarity = graycoprops(glcm, prop='dissimilarity').mean()
    homogeneity = graycoprops(glcm, prop='homogeneity').mean()
    asm = graycoprops(glcm, prop='ASM').mean()
    energy = graycoprops(glcm, prop='energy').mean()
    correlation = graycoprops(glcm, prop='correlation').mean()
    entropy = shannon_entropy(image)

    return np.array([contrast, dissimilarity, homogeneity, asm, energy, correlation, entropy])


# 移除了extract_contour_features函数，因为不再需要轮廓特征

def get_features(dataset):
    print(f"Starting feature extraction, available memory: {check_available_memory():.2f} GB")

    features = []
    labels = []

    feature_dataset = PairedDataset(
        csv_file=dataset.csv_file,  
        transform=None, 
        return_pil=True
    )

    for idx in tqdm(range(len(feature_dataset)), desc="Extracting features"):
        if idx % 100 == 0:
            mem = check_available_memory()
            print(f"Processed {idx}/{len(feature_dataset)} samples, available memory: {mem:.2f} GB")
            if mem < 1.0:
                print("Warning: Low system memory, potential crash risk!")

        clinic_image, wood_image, label = feature_dataset[idx]

        if wood_image is None or clinic_image is None:
            continue

        # Extract features - 只保留GLCM特征，移除了轮廓特征
        clinic_glcm = extract_glcm_features(clinic_image)
        wood_glcm = extract_glcm_features(wood_image)

        # Concatenate features - 只拼接GLCM特征
        concatenated_features = np.concatenate((
            clinic_glcm, 
            wood_glcm
        ))
        features.append(concatenated_features)
        labels.append(label.item())

        if idx % 500 == 0:
            gc.collect()
            print(f"Memory cleaned, available memory: {check_available_memory():.2f} GB")

    print(f"Feature extraction completed, available memory: {check_available_memory():.2f} GB")
    return np.array(features), np.array(labels)

# Calculate specificity
def calculate_specificity(y_true, y_pred):
    """Calculate specificity = TN / (TN + FP)"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size < 4:  # Ensure binary classification
        return 0.0
    tn, fp, _, _ = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# 优化的特征选择函数
def select_features(X, y, n_features=100):
    """使用方差分析(ANOVA)选择最相关的特征"""
    from sklearn.feature_selection import SelectKBest, f_classif

    print(f"Before feature selection: {X.shape[1]} features")
    selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    print(f"After feature selection: {X_selected.shape[1]} features")

    return X_selected, selector

# 主函数
def main():
    # --------------------------
    # 参数配置
    # --------------------------
    params = {
        'val_size': 0.2,          # 验证集比例
        'n_iter_search': 20,      # 增加超参数搜索迭代次数
        'cv_folds': 5,            # 交叉验证折数
        'n_jobs': 1,              # 禁用并行计算
        'sample_frac': 1.0,       # 使用全部数据
        'n_features': 150,        # 特征选择保留的特征数
        'use_feature_selection': True  # 是否使用特征选择
    }

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])

    # 内存监控
    print(f"Starting execution, total memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Available memory: {check_available_memory():.2f} GB")

    # --------------------------
    # 特征提取（执行一次）
    # --------------------------
    print("===== Starting feature extraction (executed once) =====")
    csv_file = 'data.csv'
    dataset = PairedDataset(csv_file, transform=transform, return_pil=False)
    X, y = get_features(dataset)

    # 特征选择（优化）
    if params['use_feature_selection']:
        X, selector = select_features(X, y, n_features=params['n_features'])

    # 创建特征DataFrame
    features_df = pd.DataFrame(X)
    features_df['label'] = y

    # 内存监控
    print(f"Feature data size: {features_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"Feature extraction completed, available memory: {check_available_memory():.2f} GB")

    # 释放内存
    del dataset
    gc.collect()

    # --------------------------
    # 获取所有可用模型
    # --------------------------
    try:
        # 尝试使用最新版API
        exp = ClassificationExperiment()
        exp.setup(
            data=features_df.sample(frac=0.01, random_state=42),
            target='label',
            verbose=False,
            session_id=42
        )
        all_models = exp.models()
        model_ids = all_models['ID'].dropna().tolist()
        exp = ClassificationExperiment()  # 重置实验
        print(f"Successfully retrieved {len(model_ids)} models using latest PyCaret API")
    except Exception as e:
        print(f"Failed to get models via API: {e}. Using predefined list.")
        model_ids = ['lr', 'svm', 'rf', 'xgboost', 'lightgbm', 'dt', 'knn', 'nb', 'et', 'gbc', 'catboost']

    # 确保安装了catboost
    if 'catboost' in model_ids:
        try:
            import catboost
            print("CatBoost is installed and will be included in training.")
        except ImportError:
            print("CatBoost not installed. Skipping catboost.")
            model_ids.remove('catboost')

    print(f"\n===== Found {len(model_ids)} models, training once each =====")
    print(f"Models: {', '.join(model_ids)}")

    if not model_ids:
        print("No models available. Exiting.")
        return

    # --------------------------
    # 定义优化后的超参数网格
    # --------------------------
    optimized_param_grids = {
        'lr': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2', 'elasticnet'],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [1000, 2000]
        },
        'svm': {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'degree': [2, 3, 4]  # 针对poly核
        },
        'rf': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'xgboost': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        },
        'lightgbm': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [-1, 5, 10, 20],
            'num_leaves': [31, 50, 100],
            'min_child_samples': [5, 10, 20],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'catboost': {
            'iterations': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 10],
            'border_count': [32, 64, 128]
        },
        'dt': {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1=曼哈顿距离，2=欧氏距离
        },
        'nb': {
            'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
        },
        'et': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'gbc': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
    }

    # --------------------------
    # 训练所有模型（各一次）
    # --------------------------
    all_results = {}
    failed_models = []

    for model_name in model_ids:
        print(f"\n{'='*70}")
        print(f"===== Training {model_name.upper()} (once with optimized parameters) =====")
        print(f"{'='*70}\n")

        try:
            seed = 42
            print(f"Training with seed: {seed} - {model_name.upper()}")
            print(f"Current available memory: {check_available_memory():.2f} GB")

            # 使用全部数据
            sample_df = features_df.sample(frac=params['sample_frac'], random_state=seed)

            # 初始化实验
            exp = ClassificationExperiment()

            # 设置实验
            exp.setup(
                data=sample_df,
                target='label',
                train_size=1-params['val_size'],
                normalize=True,
                normalize_method='zscore',
                fix_imbalance=True,
                fold=params['cv_folds'],
                session_id=seed,
                verbose=False,
                n_jobs=params['n_jobs']
            )

            # 添加特异性指标
            exp.add_metric(
                id='specificity',
                name='Specificity',
                score_func=calculate_specificity,
                greater_is_better=True
            )

            # 创建基础模型
            model = exp.create_model(model_name, verbose=False)

            # 超参数调优（使用优化后的网格）
            grid = optimized_param_grids.get(model_name, None)

            try:
                # 使用优化的随机搜索
                tuned_model = exp.tune_model(
                    estimator=model,
                    search_algorithm='random',
                    n_iter=params['n_iter_search'],
                    optimize='AUC',  # 优化AUC指标
                    custom_grid=grid,
                    verbose=False
                )
                print(f"Hyperparameter tuning completed for {model_name.upper()}")
            except Exception as e:
                print(f"Hyperparameter tuning failed: {e}. Using base model.")
                tuned_model = model

            # 评估模型
            exp.predict_model(tuned_model)
            metrics = exp.pull()

            # 提取关键指标（修正Precision为Prec.）
            required_metrics = ['AUC', 'Prec.', 'Recall', 'Specificity', 'Accuracy', 'F1']
            model_metrics = {}

            for metric in required_metrics:
                if metric in metrics.columns:
                    model_metrics[metric] = round(metrics[metric].iloc[0], 4)
                else:
                    print(f"Warning: Metric '{metric}' not found for {model_name}")
                    model_metrics[metric] = None

            # 保存结果
            all_results[model_name] = model_metrics
            print(f"{model_name.upper()} metrics: {model_metrics}")

            # 清理内存
            del model, tuned_model, exp, sample_df
            gc.collect()

        except Exception as e:
            print(f"Failed to train {model_name}: {str(e)[:100]}...")
            failed_models.append(model_name)
            continue

    # --------------------------
    # 整合结果为一个表格
    # --------------------------
    print(f"\n{'='*100}")
    print(f"===== All Models Training Results =====")
    print(f"Successfully trained: {len(all_results)}/{len(model_ids)} models")
    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")
    print(f"{'='*100}\n")

    # 创建结果表格
    if all_results:
        # 转换为DataFrame
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Model'

        # 按AUC排序
        results_df = results_df.sort_values(by='AUC', ascending=False)

        print("===== Model Performance Comparison (sorted by AUC) =====")
        print(results_df)

        # 保存表格为CSV
        results_df.to_csv('optimized_model_performance.csv')
        print("\nResults saved to 'optimized_model_performance.csv'")

    # --------------------------
    # 可视化：模型比较条形图
    # --------------------------
    if all_results and len(all_results) >= 2:
        plt.figure(figsize=(12, 8))
        results_df = pd.DataFrame.from_dict(all_results, orient='index')

        # 绘制主要指标条形图
        plot_metrics = ['AUC', 'Accuracy', 'F1', 'Specificity', 'Prec.', 'Recall']
        plot_data = results_df[plot_metrics].dropna(how='all')

        if not plot_data.empty:
            # 设置中文字体（避免警告）
            plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]

            ax = plot_data.plot(kind='bar', figsize=(15, 8))
            plt.title('模型性能比较（优化后）')
            plt.ylabel('得分')
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45)

            # 添加数值标签
            for p in ax.patches:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()
                ax.annotate(f'{height:.4f}', (x + width/2, y + height), ha='center', va='bottom', rotation=90)

            plt.tight_layout()
            plt.savefig('optimized_model_performance_comparison.png')
            print("Comparison plot saved to 'optimized_model_performance_comparison.png'")
            plt.close()

    print("\nAll tasks completed!")
if __name__ == "__main__":
    main()    


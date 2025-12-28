# VMTL  
**多任务框架下的稳定期与发展期白癜风判别研究**

---

## 项目描述
本项目旨在通过构建深度神经网络模型，从伍德灯及常规图像中高效提取白癜风皮损、边界、面积等关键特征，  
借助机器学习与深度学习算法进行特征学习，挖掘肉眼难以察觉的细微差异，  
将传统经验性诊断升级为数据驱动的智能诊断方式，  
以提升白癜风病期（稳定期 / 发展期）判定的准确性与效率。

---

## 目录结构

```text
vitiligo-paired/                  # 白癜风图像配对任务根目录
├─ datasets/                      # 数据集目录
│  ├─ data.csv
│  ├─ detection/                  # 目标检测数据集
│  │  ├─ classes.txt
│  │  ├─ images/
│  │  │  ├─ train/                # 含原始及数据增强后 JPG 图像
│  │  │  └─ val/
│  │  ├─ labels/
│  │  │  ├─ train/                # 含原始及数据增强后 TXT 标注
│  │  │  └─ val/
│  ├─ detection_json/             # JSON 格式检测标注
│  ├─ non-stable/                 # 非稳定期白癜风图像
│  ├─ stable/                     # 稳定期白癜风图像
│  ├─ raw/                        # 架构图、模型示意图
│  ├─ segmentation/               # 语义分割数据集
│  │  ├─ classes.txt
│  │  ├─ images/
│  │  │  ├─ train/
│  │  │  └─ val/
│  │  ├─ labels/
│  │  │  ├─ train/
│  │  │  └─ val/
│  │  ├─ labels.cache
│  │  └─ val.cache
│  └─ segmentation_json/          # JSON 格式分割标注
│
├─ outputs/                       # 模型输出目录
│  ├─ checkpoints/                # 权重存储
│  │  ├─ baseline/                # 基准模型
│  │  │  ├─ ConvNeXt/
│  │  │  ├─ PanDerm/
│  │  │  ├─ ResNet/
│  │  │  └─ ViT/
│  │  └─ proposed/                # 改进模型
│  │     ├─ best_det.pt
│  │     ├─ best_seg.pt
│  │     ├─ best_yolo_convnext_model.pth
│  │     ├─ feature.pth
│  │     ├─ pytorch_model.bin
│  │     └─ V1-V6/
│  │
│  ├─ logs/                       # 训练日志（TensorBoard）
│  │  ├─ baseline/ResNet/
│  │  └─ baseline/ViT/
│  │
│  ├─ results/                    # 实验结果
│  │  ├─ fused_images/            # 图像融合结果
│  │  │  ├─ stable/
│  │  │  └─ non-stable/
│  │  └─ runs/
│  │     ├─ detect/predict
│  │     └─ segment/predict
│  │
│  ├─ VMSL/                       # VMSL 模型输出
│  │  ├─ ablation/                # 消融实验
│  │  ├─ checkpoints/
│  │  │  └─ VMSL.pth
│  │  ├─ results/
│  │  │  ├─ predictions_resultsVMSL.csv
│  │  │  ├─ vmsl_confidence_distribution.png
│  │  │  ├─ vmsl_confusion_matrix.png
│  │  │  └─ vmsl_model_evaluation_results.csv
│  │  └─ VMSL_training_results/
│  │     ├─ v1/
│  │     ├─ v2/
│  │     ├─ v3/
│  │     └─ v4/
│  │
│  └─ YOLO/                       # YOLO 系列模型
│     ├─ checkpoints/
│     │  ├─ detection/
│     │  └─ segmentation/
│     └─ results/
│        ├─ detect/
│        │  ├─ YOLOv11/
│        │  └─ YOLOv12/
│        └─ segment/
│           ├─ YOLOv11/
│           └─ YOLOv12/
│
├─ scripts/                       # 部署与配置脚本
│  ├─ deploy.py
│  ├─ detection.yaml
│  └─ segmentation.yaml
│
├─ src/                           # 源代码
│  ├─ data/                       # 数据处理
│  ├─ evaluation/                 # 模型评估
│  ├─ models/                     # 模型定义
│  │  ├─ baseline/
│  │  ├─ VMSL/
│  │  └─ VMTL/
│  ├─ training/                   # 训练代码
│  └─ yolov12-main/
│
├─ Test/                          # 测试代码
│  ├─ VMSLV4.py
│  └─ VMTL.py
│
├─ requirements.txt
└─ README.md

# 使用说明
直接运行Test文件夹下的代码即可，其余训练预处理代码需要更改部分文件地址，对应的如何权重上传还在学习中，后续更新。
由于数据保密要求，不提供数据集。
网页部署使用的是deploy代码文件里的gradio技术。
<img width="1988" height="1399" alt="image" src="https://github.com/user-attachments/assets/9696b1a1-5d76-4fcc-b013-ad767460ead0" />
<img width="1916" height="1317" alt="image" src="https://github.com/user-attachments/assets/68fa3878-7826-448a-9966-f50b97d50b1b" />



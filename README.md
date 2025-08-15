# VMTL
多任务框架下的稳定期与发展期白癜风判别研究
# 项目描述
本项目旨在通过构建深度神经网络模型，从伍德灯及常规图像中高效提取白癜风皮损、边界、面积等关键特征，借助机器学习、深度学习算法进行特征学习，挖掘肉眼难察的细微差异，将传统经验性诊断升级为数据驱动的智能诊断，以提升白癜风病期判定的准确性与效率
# 目录结构
vitiligo-paired/  # 白癜风图像配对任务根目录
├─ datasets/  # 数据集目录
│  ├─ data.csv
│  ├─ detection/  # 目标检测数据集
│  │  ├─ classes.txt
│  │  ├─ images/
│  │  │  ├─ train/  # 含原始及数据增强后JPG图像
│  │  │  ├─ val/    # 含原始及数据增强后JPG图像
│  │  ├─ labels/
│  │  │  ├─ train/  # 含原始及数据增强后样本标注TXT文件
│  │  │  ├─ val/    # 含原始及数据增强后样本标注TXT文件
│  ├─ detection_json/  # 含JSON格式检测标注文件
│  ├─ non-stable/      # 含非稳定型白癜风JPG图像
│  ├─ raw/             # 含架构图、模型示意图等PNG文件
│  ├─ segmentation/  # 语义分割数据集
│  │  ├─ classes.txt
│  │  ├─ images/
│  │  │  ├─ train/  # 含原始及数据增强后JPG图像
│  │  │  ├─ val/    # 含原始及数据增强后JPG图像
│  │  ├─ labels/
│  │  │  ├─ train/  # 含原始及数据增强后样本标注TXT文件
│  │  │  ├─ val/    # 含原始及数据增强后样本标注TXT文件
│  │  ├─ labels.cache
│  │  ├─ val.cache
│  ├─ segmentation_json/  # 含JSON格式分割标注文件
│  ├─ stable/             # 含稳定型白癜风JPG图像
├─ outputs/  # 模型输出目录
│  ├─ checkpoints/  # 模型权重存储
│  │  ├─ baseline/  # 基准模型权重
│  │  │  ├─ ConvNeXt/：含最优模型权重文件
│  │  │  ├─ PanDerm/：含最优模型权重及训练 checkpoint 文件
│  │  │  ├─ ResNet/：含最优模型权重文件
│  │  │  ├─ ViT/：含最优模型权重文件
│  │  ├─ proposed/  # 改进模型权重
│  │  │  ├─ best_det.pt、best_seg.pt、best_yolo_convnext_model.pth、feature.pth、pytorch_model.bin
│  │  │  ├─ sam_vit_b_01ec64.pth、sam_vit_h_4b8939(1).pth
│  │  │  ├─ V1-V6/：各版本模型最优权重文件
│  ├─ logs/  # 训练日志
│  │  ├─ baseline/ResNet/：含TensorBoard格式训练日志文件
│  │  ├─ baseline/ViT/：含TensorBoard格式训练日志文件
│  ├─ results/  # 实验结果
│  │  ├─ fused_images/  # 图像融合结果
│  │  │  ├─ stable/：含稳定型样本融合JPG图像
│  │  │  ├─ non-stable/：含非稳定型样本融合JPG图像
│  │  ├─ runs/
│  │  │  ├─ detect/predict：检测任务推理结果存储目录
│  │  │  ├─ segment/predict：分割任务推理结果存储目录
│  ├─ VMSL/  # VMSL模型输出
│  │  ├─ ablation/：消融实验结果目录
│  │  ├─ checkpoints/：VMSL.pth
│  │  ├─ resluts/  # V1-V4版本结果
│  │  │  ├─ predictions_resultsVMSL.csv、vmsl_confidence_distribution_20250809_162547.png、vmsl_confusion_matrix_20250809_162513.png、vmsl_model_evaluation_results_20250809_162513.csv
│  │  │  ├─ VMSL_training_resultsv1/：best_convnext_model.pth、多轮混淆矩阵PNG文件
│  │  │  ├─ VMSL_training_resultsv2/：best_yolo_convnext_model.pth、多轮混淆矩阵PNG文件
│  │  │  ├─ VMSL_training_resultsv3/：best_yolo_convnext_model.pth、多轮混淆矩阵PNG文件
│  │  │  ├─ VMSL_training_resultsv4/：best_yolo_convnext_model.pth、多轮混淆矩阵PNG文件
│  ├─ YOLO/  # YOLO模型输出
│  │  ├─ checkpoints/  # 权重存储
│  │  │  ├─ detection/：v11best.pt、v12best.pt、yolov11n.pt、yolov12n.pt
│  │  │  ├─ segmentation/：v11best.pt、v12best.pt、yolo11n-seg.pt、yolov12n-seg.pt
│  │  ├─ resluts/  # 结果存储
│  │  │  ├─ detect/
│  │  │  │  ├─ YOLOV11/：args.yaml、confusion_matrix.png、confusion_matrix_normalized.png、F1_curve.png、labels.jpg、labels_correlogram.jpg、PR_curve.png、P_curve.png、results.csv、results.png、R_curve.png、多批次训练/验证可视化JPG、weights/last.pt
│  │  │  │  ├─ YOLOV12/：args.yaml、confusion_matrix.png、confusion_matrix_normalized.png、detection_result.jpg、F1_curve.png、labels.jpg、labels_correlogram.jpg、PR_curve.png、P_curve.png、results.csv、results.png、R_curve.png、多批次训练/验证可视化JPG、weights/best.pt、weights/last.pt
│  │  │  ├─ segment/
│  │  │  │  ├─ YOLOV11/：args.yaml、BoxF1_curve.png、BoxPR_curve.png、BoxP_curve.png、BoxR_curve.png、confusion_matrix.png、confusion_matrix_normalized.png、labels.jpg、labels_correlogram.jpg、MaskF1_curve.png、MaskPR_curve.png、MaskP_curve.png、MaskR_curve.png、results.csv、results.png、多批次训练/验证可视化JPG、weights/last.pt
│  │  │  │  ├─ YOLOV12/：args.yaml、BoxF1_curve.png、BoxPR_curve.png、BoxP_curve.png、BoxR_curve.png、confusion_matrix.png、confusion_matrix_normalized.png、labels.jpg、labels_correlogram.jpg、MaskF1_curve.png、MaskPR_curve.png、MaskP_curve.png、MaskR_curve.png、results.csv、results.png、segmentation_result.jpg、多批次训练/验证可视化JPG、weights/best.pt、weights/last.pt
├─ scripts/  # 脚本目录
│  ├─ .gradio/：certificate.pem
│  ├─ deploy.py
│  ├─ detection.yaml
│  ├─ segmentation.yaml
├─ src/  # 源代码目录
│  ├─ data/  # 数据处理代码
│  │  ├─ Data_set.ipynb
│  │  ├─ feature.ipynb
│  │  ├─ kuozeng.py
│  │  ├─ segmentation.py
│  │  ├─ 将json转为yolo.ipynb
│  │  ├─ 数据集处理.ipynb
│  ├─ evaluation/：evaluate_modelsVMSL.py
│  ├─ models/  # 模型定义代码
│  │  ├─ baseline/
│  │  │  ├─ dl_model/：panderm.py、ViT_ConvNeXt_ResNet.py
│  │  │  ├─ ml_models/：v1_v4.py、v2_v3.py
│  │  ├─ ssPOC/：sparse.ipynb
│  │  ├─ VMSL/：detection.py、predict_both_modelsVMSL.py
│  │  ├─ VMTL/：Cross Attention and Data Augmentation(V3).ipynb、Multi-task Learning(V5).py、Mutiple-Stage(V4).ipynb、Simple attention and MAE(V1).ipynb、Simple Attention, MAE, and Data-Augmentation(V2).ipynb、VMTL.py
│  ├─ training/  # 训练代码
│  │  ├─ feature_extracted_visual.ipynb
│  │  ├─ VMSLV1.py
│  │  ├─ VMSLV2.py
│  │  ├─ VMSLV3.py
│  │  ├─ VMSLV4.py
│  │  ├─ xiaobo.ipynb
│  ├─ yolov12-main/  # YOLOv12核心目录
├─ ~Test~/  # 测试目录
│  ├─ VMSLV4.py
│  ├─ VMTL.py
├─ README.md.txt
├─ requirements.txt

# 使用说明
直接运行~Test~文件夹下的代码即可，其余训练预处理代码需要更改部分文件地址，对应的如何权重上传还在学习中，后续更新

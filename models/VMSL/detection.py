from ultralytics import YOLO


if __name__ == '__main__':
  model = YOLO('yolov11n.pt')
  # Train the model
  results = model.train(
    data='detection.yaml',
    epochs=1200, 
    batch=16, 
    imgsz=640,
    patience=100,
    scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
    mosaic=1.0,
    mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
    copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
    device="0",
  )

  
'''results = model.train(
  data='detection.yaml',
  epochs=600, 
  batch=16, 
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0",
  dropout=0.5,
  lr0=0.0001,
  weight_decay=0.0005,
  cos_lr=True,
  patience=200,           # Early stopping patience 
  flipud=0.1,             # 大幅降低上下翻转（从0.5→0.1），皮肤部位上下翻转意义低
  fliplr=0.5,             # 保持左右翻转（不影响病变形态）
  degrees=7,              # 降低旋转角度（从10→7），避免病变边界扭曲
  translate=0.1,          # 保持轻微平移，增加位置多样性
  shear=2.0,              # 降低剪切幅度（从3.0→2.0），减少病变形状失真
  hsv_h=0.015,            # 大幅降低色相调整（从0.5→0.015），避免肤色失真
  hsv_s=0.3,              # 降低饱和度调整（从0.4→0.3），保留病变与正常皮肤的色差
  hsv_v=0.4,              # 提高明度调整（从0.2→0.4），适应不同光照下的病变（如逆光/强光）
  erasing=0.05 
)'''

'''results = model.train(
    data='detection.yaml',  # 确保yaml中包含白癜风数据集的训练/验证路径
    epochs=100,             # 100轮足够，配合早停避免过拟合
    batch=16,               # 若GPU显存不足可降为8（如RTX 4060 Laptop建议8）
    imgsz=640,              # 640适合捕捉中小病变区域细节
    patience=100,           # 给予充足耐心，避免早停

    # 数据增广优化（核心：增强多样性且不破坏病变特征）
    scale=0.3,              # 降低缩放幅度（从0.5→0.3），避免小病变拉伸变形
    mosaic=0.6,             # 新增马赛克增强（0.6概率），融合4张图增加场景多样性
    copy_paste=0.2,         # 降低复制粘贴概率（从0.3→0.2），避免病变堆叠不自然
    flipud=0.1,             # 大幅降低上下翻转（从0.5→0.1），皮肤部位上下翻转意义低
    fliplr=0.5,             # 保持左右翻转（不影响病变形态）
    degrees=7,              # 降低旋转角度（从10→7），避免病变边界扭曲
    translate=0.1,          # 保持轻微平移，增加位置多样性
    shear=2.0,              # 降低剪切幅度（从3.0→2.0），减少病变形状失真
    hsv_h=0.015,            # 大幅降低色相调整（从0.5→0.015），避免肤色失真
    hsv_s=0.3,              # 降低饱和度调整（从0.4→0.3），保留病变与正常皮肤的色差
    hsv_v=0.4,              # 提高明度调整（从0.2→0.4），适应不同光照下的病变（如逆光/强光）
    erasing=0.05,           # 大幅降低擦除概率（从0.3→0.05），避免关键病变区域被擦除

    # 模型优化（适配细粒度特征学习）
    dropout=0.4,            # 降低dropout（从0.7→0.4），避免丢失模糊病变的细节特征
    weight_decay=0.0005,    # 提高权重衰减，抑制过拟合（白癜风样本可能类别不平衡）
    lr0=0.001,              # 略提高初始学习率（从0.0008→0.001），加速收敛
    cos_lr=True,            # 保持余弦学习率，后期精细化优化
    warmup_epochs=10,       # 延长热身轮次（从8→10），让模型先适应皮肤基础特征
    optimizer='Adam',       # 适合细粒度特征学习

    # 硬件配置
    device="0",
    workers=8,              # 降低线程数（从10→8），避免CPU-GPU数据传输瓶颈
    amp=True,               # 保持混合精度，节省显存
)
'''
''' results = model.train(
    data='detection.yaml',  # 原始数据集配置（无需新增文件）
    epochs=100,
    batch=16,
    imgsz=640,
    patience=100,
    scale=0.5,
    copy_paste=0.3,
    flipud=0.5,
    fliplr=0.5,
    degrees=10,
    translate=0.1,
    shear=3.0,
    hsv_h=0.5,
    hsv_s=0.4,
    hsv_v=0.2,
    erasing=0.3,

    dropout=0.7,           
    weight_decay=0.0003,   
    lr0=0.0008,            
    cos_lr=True,           
    warmup_epochs=8,       
    optimizer='Adam',      
    device="0",            
    workers=10,            
    amp=True,              
)'''
    

'''  results = model.train(
    data='detection.yaml',  # Path to the dataset configuration file
    epochs=1000,            # Number of training epochs    
    batch=16,             # Batch size for training
    imgsz=640,            # Input image size for training

    #数据增广
    scale=0.7,      
    mosaic=1.0,     
    mixup=0.2,      
    copy_paste=0.3, 
    flipud=0.2,     
    fliplr=0.5,     
    degrees=10.0,   
    perspective=0.001,
    shear=5.0,      
    hsv_h=0.015,    
    hsv_s=0.7,      
    hsv_v=0.4,      
    erasing=0.2,    

    dropout=0.7,
    lr0 = 0.0005,
    patience=100,  # Early stopping patience
    device="0",    # Specify the device to use for training (e.g., "0" for GPU 0, "cpu" for CPU)
)'''
    


  # # Evaluate model performance on the validation set
  # metrics = model.val()

  # # Perform object detection on an image
  # results = model("path/to/image.jpg")
  # results[0].show()


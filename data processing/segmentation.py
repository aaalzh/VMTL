from ultralytics import YOLO


if __name__ == '__main__':
  model = YOLO('yolo11n-seg.pt')
  results = model.train(
    data='segmentation.yaml',
    epochs=1000,
    batch=16, 
    imgsz=640,
    patience=100,
    scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
    mosaic=1.0,
    mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
    copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
    device="0",
    dropout=0.5,
    lr0=0.00005,  # Lower learning rate for better convergence
)


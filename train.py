from ultralytics import YOLO
import torch

if __name__ == "__main__":

    """
    # 加载预训练的YOLOv8n检测模型
    model = YOLO("yolov8n.pt")  # 使用检测模型

    # 设置设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 训练模型
    train_results = model.train(
        data="./NaiLong.v5i.yolov8-obb/data.yaml",  # 使用YAML配置文件
        epochs=100,
        batch=16,  # 批大小
        imgsz=640,
        device=device,  # 使用GPU
        workers=0,  # Windows上设为0避免多进程问题
        name="whole_image_detection",
    )

    # 评估模型
    metrics = model.val()
    """
    model = YOLO("runs/detect/whole_image_detection/weights/best.pt")
    # 导出模型
    path = model.export(format="onnx", opset=15, dynamic=True)
    print(f"模型已导出至: {path}")

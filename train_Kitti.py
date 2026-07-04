from ultralytics import YOLO


# model = YOLO('yolo11n.yaml')            # Baseline
# model = YOLO('HSSE_YOLO-n.yaml')        # 完整 HSSE-YOLO


model = YOLO('yolo11n.yaml')


def main():
    model.train(
        data='kitti_data.yaml',
        epochs=300,
        imgsz=640,
        batch=32,
        device='0',
        patience=50,
    )

if __name__ == '__main__':
    main()

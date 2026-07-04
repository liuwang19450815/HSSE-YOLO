from ultralytics import YOLO


# model = YOLO('yolo11n.yaml')            # Baseline
# model = YOLO('HSSE_YOLO-n.yaml')        # 完整 HSSE-YOLO


model = YOLO('HSSE_YOLO-n.yaml')


def main():
    model.train(
        data='CCTSDB_data.yaml',
        epochs=300,
        imgsz=800,  # HSSE-YOLO
        batch=32,
        device='0',
        patience=50,
    )

if __name__ == '__main__':
    main()

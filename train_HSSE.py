from ultralytics import YOLO

# ============================================================
# 消融实验 — 取消注释要跑的那一行
# ============================================================

# model = YOLO('yolo11n.yaml')            # Baseline
# model = YOLO('ODCA-n.yaml')             # 单模块
# model = YOLO('P2L_Head-n.yaml')
# model = YOLO('BAW_Concat-n.yaml')
# model = YOLO('ODCA_BAW-n.yaml')         # 双模块 — 需补
# model = YOLO('ODCA_P2L_Head-n.yaml')
# model = YOLO('P2L_Head_BAW-n.yaml')     # 双模块 — 需补
# model = YOLO('HSSE_YOLO-n.yaml')        # 完整 HSSE-YOLO
model = YOLO('ODCA_BAW-n.yaml')


def main():
    model.train(
        data='HSSE_data.yaml',
        epochs=300,
        imgsz=800,
        batch=32,
        device='0',
        patience=50,
    )

if __name__ == '__main__':
    main()

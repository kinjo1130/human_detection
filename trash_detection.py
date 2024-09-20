from ultralytics import YOLO
import cv2
import numpy as np
import time

# YOLOv8モデルの読み込み
model = YOLO('yolov8n.pt')

# ビデオの読み込み（ここではカメラを使用。ファイルを使用する場合は引数をファイルパスに変更）
cap = cv2.VideoCapture(0)

# カメラの設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# YOLOv8の描画オプションを設定
options = {
    "boxes": True,
    "conf": True,
    "labels": True,
    "line_width": 2,
    "font_size": 1,
    "font": 'Arial.ttf',
    "color": (0, 255, 0),
}

def apply_overlay(frame, boxes, color=(255, 0, 0), alpha=0.3):
    overlay = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

prev_time = 0
while cap.isOpened():
    current_time = time.time()
    if current_time - prev_time < 1/30:  # 30 FPSに制限
        continue
    prev_time = current_time

    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break

    try:
        # YOLOv8による検出の実行（人物クラスのみ）
        results = model(frame, classes=[0], conf=0.5)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # 人物検出領域に青色のオーバーレイを追加
            boxes = results[0].boxes.xyxy.cpu().numpy()
            frame = apply_overlay(frame, boxes)

            # バウンディングボックスとラベルを描画
            annotated_frame = results[0].plot(**options)
        else:
            annotated_frame = frame

        # 結果の表示
        cv2.imshow("YOLOv8 Person Detection", annotated_frame)

    except Exception as e:
        print(f"An error occurred: {e}")
        continue

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
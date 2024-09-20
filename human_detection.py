from ultralytics import YOLO
import cv2
import numpy as np
# YOLOv8モデルの読み込み
model = YOLO('yolov8n.pt')
# ビデオの読み込み（ここではカメラを使用。ファイルを使用する場合は引数をファイルパスに変更）
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # フレームの読み込み
    success, frame = cap.read()
    if success:
        # YOLOv8による検出の実行
        results = model(frame)
        # 検出結果の描画
        annotated_frame = results[0].plot()
        # 結果の表示
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
# リソースの解放
cap.release()
cv2.destroyAllWindows()
import cv2
import time
from ultralytics import YOLO

# YOLOモデルの読み込み（CPUモードで）
person_model = YOLO('yolov8n.pt').to('cpu')
trash_model = YOLO('../human-detect/runs/detect/train5/weights/best.pt').to('cpu')

# カメラの設定
cap = cv2.VideoCapture(0)

# 検出のための設定
conf_threshold = 0.5  # 信頼度のしきい値

def draw_detections(frame, detections, color):
    for r in detections:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{r.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

try:
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # フレームをリサイズ
        frame = cv2.resize(frame, (640, 480))

        # 人物検出を実行
        person_results = person_model(frame, conf=conf_threshold, classes=[0])

        # ゴミ検出を実行
        trash_results = trash_model(frame, conf=conf_threshold)

        # 検出結果の描画
        draw_detections(frame, person_results, (255, 0, 0))  # 青色
        draw_detections(frame, trash_results, (0, 255, 0))   # 緑色

        # FPSの計算と表示
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 結果を表示
        cv2.imshow("Person and Trash Detection", frame)

        # キー入力を待つ（30ミリ秒）
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / elapsed_time:.2f}")
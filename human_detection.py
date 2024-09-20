from ultralytics import YOLO
import cv2

# YOLOv8モデルの読み込み
model = YOLO('yolov8n.pt')

# ビデオの読み込み（ここではカメラを使用。ファイルを使用する場合は引数をファイルパスに変更）
cap = cv2.VideoCapture(0)

# YOLOv8の描画オプションを設定
options = {
    "boxes": True,  # バウンディングボックスを表示
    "conf": True,   # 信頼度スコアを表示
    "labels": True, # ラベルを表示
    "line_width": 2,  # 線の太さ
    "font_size": 1,   # フォントサイズ
    "font": 'Arial.ttf',  # フォント

}

while cap.isOpened():
    # フレームの読み込み
    success, frame = cap.read()

    if success:
        # YOLOv8による検出の実行と結果の描画
        results = model.predict(frame, classes=[0], conf=0.5, verbose=False)[0]  # class 0 は 'person'
        annotated_frame = results.plot(**options)

        # 人物検出領域に青色のオーバーレイを追加
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), -1)  # 青色で塗りつぶし
            cv2.addWeighted(annotated_frame, 0.5, frame, 0.5, 0, annotated_frame)  # 半透明化

        # 結果の表示
        cv2.imshow("YOLOv8 Person Detection", annotated_frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
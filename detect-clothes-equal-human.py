import cv2
import numpy as np
import os
import logging
import glob
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# YOLOv8モデルの読み込み
yolo_model = YOLO('yolov8n.pt')

def extract_clothing_features(img):
    results = yolo_model(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if r.names[cls] == 'person':
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 服の領域を推定（上半身全体を対象に）
                height = y2 - y1
                clothing_y1 = y1 + height // 8  # 上端を少し上げる
                clothing_y2 = y1 + height // 2 + height // 4  # 下端を少し下げる
                clothing_region = img[clothing_y1:clothing_y2, x1:x2]
                
                if clothing_region.size == 0:
                    continue
                
                # RGBの中央値を計算
                rgb_median = np.median(clothing_region, axis=(0, 1))
                
                # HSVに変換
                hsv_region = cv2.cvtColor(clothing_region, cv2.COLOR_BGR2HSV)
                hsv_median = np.median(hsv_region, axis=(0, 1))
                
                return np.concatenate([rgb_median, hsv_median]), (x1, clothing_y1, x2, clothing_y2), clothing_region
    
    return None, None, None

def analyze_clothing(img_path):
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"'{img_path}' を読み込めませんでした。")
        return None

    features, clothing_region, clothing_img = extract_clothing_features(img)
    if features is None:
        logging.warning(f"'{os.path.basename(img_path)}' から服の領域を検出できませんでした。")
        return None

    # 分析結果を可視化
    plt.figure(figsize=(15, 10))

    # 元の画像を表示
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # 検出された服の領域を表示
    plt.subplot(2, 3, 2)
    x1, y1, x2, y2 = clothing_region
    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB))
    plt.title("Detected Clothing Region")
    plt.axis('off')

    # 服の領域のみを表示
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(clothing_img, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Clothing")
    plt.axis('off')

    # RGBヒストグラムを表示
    plt.subplot(2, 3, 4)
    color = ('b','g','r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([clothing_img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
    plt.title("RGB Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # HSVヒストグラムを表示
    plt.subplot(2, 3, 5)
    hsv_img = cv2.cvtColor(clothing_img, cv2.COLOR_BGR2HSV)
    plt.hist(hsv_img[:,:,0].ravel(), 180, [0, 180], color='r', alpha=0.5, label="Hue")
    plt.hist(hsv_img[:,:,1].ravel(), 256, [0, 256], color='g', alpha=0.5, label="Saturation")
    plt.hist(hsv_img[:,:,2].ravel(), 256, [0, 256], color='b', alpha=0.5, label="Value")
    plt.legend()
    plt.title("HSV Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # 代表色を表示
    plt.subplot(2, 3, 6)
    dominant_color = features[:3].astype(int)
    color_patch = np.full((100, 100, 3), dominant_color, dtype=np.uint8)
    plt.imshow(color_patch)
    plt.title(f"Dominant Color: RGB{tuple(dominant_color)}")
    plt.axis('off')

    # 画像を保存
    output_path = os.path.join('analysis_results', f"analysis_{os.path.basename(img_path)}")
    os.makedirs('analysis_results', exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logging.info(f"分析結果を保存しました: {output_path}")
    return features

def load_target_images(image_dir):
    target_features = []
    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_files:
        logging.error(f"ディレクトリ '{image_dir}' に画像が見つかりません。")
        return None

    for image_path in image_files:
        features = analyze_clothing(image_path)
        if features is not None:
            target_features.append(features)

    if not target_features:
        logging.error("いずれの画像からも服の領域を検出できませんでした。")
        return None

    return target_features

def color_distance(color1, color2):
    return np.sqrt(np.sum((color1 - color2)**2))

def detect_specific_person(frame, target_features, threshold=50):
    frame_features, clothing_region, _ = extract_clothing_features(frame)
    if frame_features is None:
        return False, None, None, float('inf')
    
    min_distance = float('inf')
    for i, target in enumerate(target_features):
        distance = color_distance(frame_features, target)
        if distance < min_distance:
            min_distance = distance
        if distance < threshold:
            return True, frame_features, clothing_region, min_distance
    
    return False, frame_features, clothing_region, min_distance

def take_photo(frame, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"detected_person_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    logging.info(f"写真を保存しました: {filename}")

def main():
    target_image_dir = 'target_persons'
    output_image_dir = 'images'
    target_features = load_target_images(target_image_dir)
    if target_features is None:
        logging.error("ターゲット画像の読み込みに失敗したため、プログラムを終了します。")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    person_found = False

    logging.info("カメラからの映像取得を開始します。")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logging.error("フレームの取得に失敗しました。")
            break

        person_detected, detected_features, clothing_region, min_distance = detect_specific_person(frame, target_features)

        if clothing_region:
            x1, y1, x2, y2 = clothing_region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if person_detected:
            cv2.putText(frame, "Target Person", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            take_photo(frame, output_image_dir)
            person_found = True
            break

        # 類似度（距離）のログを出力
        logging.info(f"最小色距離: {min_distance:.2f}")

        cv2.imshow("Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or person_found:
            logging.info("プログラムを終了します。")
            break

    cap.release()
    cv2.destroyAllWindows()

    if person_found:
        logging.info("特定の人物を検出し、写真を撮影しました。")
    else:
        logging.info("特定の人物を検出できませんでした。")

if __name__ == "__main__":
    main()
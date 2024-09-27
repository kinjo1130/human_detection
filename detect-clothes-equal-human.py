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
                
                height = y2 - y1
                # 上半身の領域
                upper_y1 = y1
                upper_y2 = y1 + height // 2
                upper_region = img[upper_y1:upper_y2, x1:x2]
                
                # 下半身の領域
                lower_y1 = y1 + height // 2
                lower_y2 = y2
                lower_region = img[lower_y1:lower_y2, x1:x2]
                
                if upper_region.size == 0 or lower_region.size == 0:
                    continue
                
                # 上半身のRGBとHSVの中央値を計算
                upper_rgb_median = np.median(upper_region, axis=(0, 1))
                upper_hsv_region = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
                upper_hsv_median = np.median(upper_hsv_region, axis=(0, 1))
                
                # 下半身のRGBとHSVの中央値を計算
                lower_rgb_median = np.median(lower_region, axis=(0, 1))
                lower_hsv_region = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)
                lower_hsv_median = np.median(lower_hsv_region, axis=(0, 1))
                
                upper_features = np.concatenate([upper_rgb_median, upper_hsv_median])
                lower_features = np.concatenate([lower_rgb_median, lower_hsv_median])
                
                return upper_features, lower_features, (x1, upper_y1, x2, upper_y2), (x1, lower_y1, x2, lower_y2)
    
    return None, None, None, None

def analyze_clothing(img_path):
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"'{img_path}' を読み込めませんでした。")
        return None, None

    upper_features, lower_features, upper_region, lower_region = extract_clothing_features(img)
    if upper_features is None or lower_features is None:
        logging.warning(f"'{os.path.basename(img_path)}' から服の領域を検出できませんでした。")
        return None, None

    # 分析結果を可視化
    plt.figure(figsize=(15, 10))

    # 元の画像を表示
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # 検出された服の領域を表示
    plt.subplot(2, 3, 2)
    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect, (upper_region[0], upper_region[1]), (upper_region[2], upper_region[3]), (0, 255, 0), 2)
    cv2.rectangle(img_with_rect, (lower_region[0], lower_region[1]), (lower_region[2], lower_region[3]), (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB))
    plt.title("Detected Clothing Regions")
    plt.axis('off')

    # 上半身の服の領域のみを表示
    plt.subplot(2, 3, 3)
    upper_clothing = img[upper_region[1]:upper_region[3], upper_region[0]:upper_region[2]]
    plt.imshow(cv2.cvtColor(upper_clothing, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Upper Clothing")
    plt.axis('off')

    # 下半身の服の領域のみを表示
    plt.subplot(2, 3, 4)
    lower_clothing = img[lower_region[1]:lower_region[3], lower_region[0]:lower_region[2]]
    plt.imshow(cv2.cvtColor(lower_clothing, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Lower Clothing")
    plt.axis('off')

    # 上半身の代表色を表示
    plt.subplot(2, 3, 5)
    upper_dominant_color = upper_features[:3].astype(int)
    upper_color_patch = np.full((100, 100, 3), upper_dominant_color, dtype=np.uint8)
    plt.imshow(upper_color_patch)
    plt.title(f"Upper Dominant Color: RGB{tuple(upper_dominant_color)}")
    plt.axis('off')

    # 下半身の代表色を表示
    plt.subplot(2, 3, 6)
    lower_dominant_color = lower_features[:3].astype(int)
    lower_color_patch = np.full((100, 100, 3), lower_dominant_color, dtype=np.uint8)
    plt.imshow(lower_color_patch)
    plt.title(f"Lower Dominant Color: RGB{tuple(lower_dominant_color)}")
    plt.axis('off')

    # 画像を保存
    output_path = os.path.join('analysis_results', f"analysis_{os.path.basename(img_path)}")
    os.makedirs('analysis_results', exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logging.info(f"分析結果を保存しました: {output_path}")
    return upper_features, lower_features

def load_target_images(image_dir):
    target_features = []
    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_files:
        logging.error(f"ディレクトリ '{image_dir}' に画像が見つかりません。")
        return None

    for image_path in image_files:
        upper_features, lower_features = analyze_clothing(image_path)
        if upper_features is not None and lower_features is not None:
            target_features.append((upper_features, lower_features))
            logging.info(f"'{os.path.basename(image_path)}' から服の特徴量を抽出しました。")
        else:
            logging.warning(f"'{os.path.basename(image_path)}' から服の特徴量を抽出できませんでした。")

    if not target_features:
        logging.error("いずれの画像からも服の領域を検出できませんでした。")
        return None

    return target_features

def color_distance(color1, color2):
    return np.sqrt(np.sum((color1 - color2)**2))

def detect_specific_person(frame, target_features, threshold=50):
    frame_upper_features, frame_lower_features, upper_region, lower_region = extract_clothing_features(frame)
    if frame_upper_features is None or frame_lower_features is None:
        return False, None, None, float('inf')
    
    min_distance = float('inf')
    for target_upper, target_lower in target_features:
        upper_distance = color_distance(frame_upper_features, target_upper)
        lower_distance = color_distance(frame_lower_features, target_lower)
        total_distance = (upper_distance + lower_distance) / 2
        
        if total_distance < min_distance:
            min_distance = total_distance
        
        if total_distance < threshold:
            return True, (frame_upper_features, frame_lower_features), (upper_region, lower_region), min_distance
    
    return False, (frame_upper_features, frame_lower_features), (upper_region, lower_region), min_distance

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

        person_detected, detected_features, clothing_regions, min_distance = detect_specific_person(frame, target_features)

        if clothing_regions:
            upper_region, lower_region = clothing_regions
            cv2.rectangle(frame, (upper_region[0], upper_region[1]), (upper_region[2], upper_region[3]), (0, 255, 0), 2)
            cv2.rectangle(frame, (lower_region[0], lower_region[1]), (lower_region[2], lower_region[3]), (0, 0, 255), 2)

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
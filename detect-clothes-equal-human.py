import cv2
import numpy as np
import os
import logging
import glob
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# YOLOv8モデルの読み込み (セグメンテーションモデルを使用)
yolo_model = YOLO('yolov8n-seg.pt')

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[2]), int(rgb[1]), int(rgb[0]))  # OpenCVはBGR順なので注意

def hsv_to_rgb(hsv):
    return cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0]

def extract_color_features(img, mask):
    # マスクを適用して背景を除外
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # RGBヒストグラム
    rgb_hist = cv2.calcHist([masked_img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    rgb_hist = cv2.normalize(rgb_hist, rgb_hist).flatten()

    # HSV変換とヒストグラム
    hsv_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv_img], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    # RGB中央値 (マスクされた領域のみ)
    rgb_median = np.median(masked_img[mask > 0], axis=0).astype(int)

    # HSV中央値 (マスクされた領域のみ)
    hsv_median = np.median(hsv_img[mask > 0], axis=0).astype(int)

    return rgb_hist, hsv_hist, rgb_median, hsv_median, masked_img

def extract_clothing_features(img):
    results = yolo_model(img)
    
    # 検出結果が空の場合、早期リターン
    if len(results) == 0 or len(results[0].boxes) == 0:
        logging.warning("人物が検出されませんでした。")
        return None, None, None, None

    for r in results:
        masks = r.masks
        boxes = r.boxes
        
        # マスクまたはボックスが存在しない場合、次のイテレーションへ
        if masks is None or boxes is None or len(masks) == 0 or len(boxes) == 0:
            logging.warning("マスクまたはボックスが存在しません。")
            continue

        for mask, box in zip(masks, boxes):
            cls = int(box.cls[0])
            if r.names[cls] == 'person':
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # セグメンテーションマスクを取得
                segment_mask = mask.data[0].cpu().numpy()
                segment_mask = (segment_mask * 255).astype(np.uint8)
                segment_mask = cv2.resize(segment_mask, (img.shape[1], img.shape[0]))

                height = y2 - y1
                # 上半身の領域
                upper_y1 = y1
                upper_y2 = y1 + height // 2
                upper_mask = segment_mask[upper_y1:upper_y2, x1:x2]
                upper_region = img[upper_y1:upper_y2, x1:x2]
                
                # 下半身の領域
                lower_y1 = y1 + height // 2
                lower_y2 = y2
                lower_mask = segment_mask[lower_y1:lower_y2, x1:x2]
                lower_region = img[lower_y1:lower_y2, x1:x2]
                
                if upper_region.size == 0 or lower_region.size == 0:
                    continue
                
                # 上半身と下半身の特徴を抽出
                upper_features = extract_color_features(upper_region, upper_mask)
                lower_features = extract_color_features(lower_region, lower_mask)
                
                return upper_features, lower_features, (x1, upper_y1, x2, upper_y2), (x1, lower_y1, x2, lower_y2)
    
    logging.warning("適切な人物の服装領域が検出されませんでした。")
    return None, None, None, None
# analyze_clothing 関数の変更 (主に可視化部分)
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
    plt.figure(figsize=(20, 15))

    # 元の画像を表示
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # 検出された服の領域を表示
    plt.subplot(3, 3, 2)
    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect, (upper_region[0], upper_region[1]), (upper_region[2], upper_region[3]), (0, 255, 0), 2)
    cv2.rectangle(img_with_rect, (lower_region[0], lower_region[1]), (lower_region[2], lower_region[3]), (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB))
    plt.title("Detected Clothing Regions")
    plt.axis('off')

    # 上半身の服の領域のみを表示 (マスク適用後)
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(upper_features[4], cv2.COLOR_BGR2RGB))
    plt.title("Extracted Upper Clothing (Masked)")
    plt.axis('off')

    # 下半身の服の領域のみを表示 (マスク適用後)
    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(lower_features[4], cv2.COLOR_BGR2RGB))
    plt.title("Extracted Lower Clothing (Masked)")
    plt.axis('off')

    # 上半身のRGBヒストグラムを表示
    plt.subplot(3, 3, 5)
    plt.bar(range(len(upper_features[0])), upper_features[0])
    plt.title("Upper Clothing RGB Histogram")
    plt.xlabel("Bin")
    plt.ylabel("Normalized Frequency")

    # 下半身のRGBヒストグラムを表示
    plt.subplot(3, 3, 6)
    plt.bar(range(len(lower_features[0])), lower_features[0])
    plt.title("Lower Clothing RGB Histogram")
    plt.xlabel("Bin")
    plt.ylabel("Normalized Frequency")

    # 上半身のHSVヒストグラムを表示
    plt.subplot(3, 3, 7)
    plt.bar(range(len(upper_features[1])), upper_features[1])
    plt.title("Upper Clothing HSV Histogram")
    plt.xlabel("Bin")
    plt.ylabel("Normalized Frequency")

    # 下半身のHSVヒストグラムを表示
    plt.subplot(3, 3, 8)
    plt.bar(range(len(lower_features[1])), lower_features[1])
    plt.title("Lower Clothing HSV Histogram")
    plt.xlabel("Bin")
    plt.ylabel("Normalized Frequency")

    # RGB中央値とHSV中央値をカラーコードで表示
    plt.subplot(3, 3, 9)
    ax = plt.gca()
    ax.axis('off')
    plt.title("Color Statistics (Median)")

    # 上半身のRGB中央値
    upper_rgb_hex = rgb_to_hex(upper_features[2])
    ax.add_patch(patches.Rectangle((0.1, 0.8), 0.1, 0.1, facecolor=upper_rgb_hex))
    plt.text(0.22, 0.85, f"Upper RGB Median: {upper_rgb_hex}", fontsize=10, verticalalignment='center')

    # 上半身のHSV中央値（RGBに変換）
    upper_hsv_rgb = hsv_to_rgb(upper_features[3])
    upper_hsv_hex = rgb_to_hex(upper_hsv_rgb)
    ax.add_patch(patches.Rectangle((0.1, 0.65), 0.1, 0.1, facecolor=upper_hsv_hex))
    plt.text(0.22, 0.7, f"Upper HSV Median: {upper_hsv_hex}", fontsize=10, verticalalignment='center')

    # 下半身のRGB中央値
    lower_rgb_hex = rgb_to_hex(lower_features[2])
    ax.add_patch(patches.Rectangle((0.1, 0.5), 0.1, 0.1, facecolor=lower_rgb_hex))
    plt.text(0.22, 0.55, f"Lower RGB Median: {lower_rgb_hex}", fontsize=10, verticalalignment='center')

    # 下半身のHSV中央値（RGBに変換）
    lower_hsv_rgb = hsv_to_rgb(lower_features[3])
    lower_hsv_hex = rgb_to_hex(lower_hsv_rgb)
    ax.add_patch(patches.Rectangle((0.1, 0.35), 0.1, 0.1, facecolor=lower_hsv_hex))
    plt.text(0.22, 0.4, f"Lower HSV Median: {lower_hsv_hex}", fontsize=10, verticalalignment='center')

    # 画像を保存
    output_path = os.path.join('analysis_results', f"analysis_{os.path.basename(img_path)}")
    os.makedirs('analysis_results', exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logging.info(f"分析結果を保存しました: {output_path}")
    return upper_features[:4], lower_features[:4]  # マスク適用後の画像は返さない

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

def feature_similarity(features1, features2):
    if features1 is None or features2 is None:
        return 0  # 類似度が計算できない場合は0を返す

    rgb_hist_sim = cv2.compareHist(features1[0], features2[0], cv2.HISTCMP_CORREL)
    hsv_hist_sim = cv2.compareHist(features1[1], features2[1], cv2.HISTCMP_CORREL)
    rgb_median_sim = 1 - np.linalg.norm(features1[2] - features2[2]) / 255
    hsv_median_sim = 1 - np.linalg.norm(features1[3] - features2[3]) / 255
    return (rgb_hist_sim + hsv_hist_sim + rgb_median_sim + hsv_median_sim) / 4

def detect_specific_person(frame, target_features, threshold=0.65):
    result = extract_clothing_features(frame)
    if result is None:
        return False, None, None, 0
    
    frame_upper_features, frame_lower_features, upper_region, lower_region = result

    max_similarity = 0
    for target_upper, target_lower in target_features:
        if frame_upper_features is None or frame_lower_features is None:
            continue  # 特徴量が抽出できなかった場合はスキップ

        upper_similarity = feature_similarity(frame_upper_features, target_upper)
        lower_similarity = feature_similarity(frame_lower_features, target_lower)
        total_similarity = (upper_similarity + lower_similarity) / 2

        if total_similarity > max_similarity:
            max_similarity = total_similarity

        if total_similarity > threshold:
            return True, (frame_upper_features, frame_lower_features), (upper_region, lower_region), max_similarity

    return False, (frame_upper_features, frame_lower_features), (upper_region, lower_region), max_similarity

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
            continue

        person_detected, detected_features, clothing_regions, max_similarity = detect_specific_person(frame, target_features)

        if clothing_regions is not None:
            print(clothing_regions)
            upper_region, lower_region = clothing_regions
            if upper_region is not None and lower_region is not None:
                cv2.rectangle(frame, (upper_region[0], upper_region[1]), (upper_region[2], upper_region[3]), (0, 255, 0), 2)
            if lower_region is not None:
                cv2.rectangle(frame, (lower_region[0], lower_region[1]), (lower_region[2], lower_region[3]), (0, 0, 255), 2)

        if person_detected:
            cv2.putText(frame, "Target Person", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            take_photo(frame, output_image_dir)
            person_found = True
            break

        # 類似度のログを出力
        logging.info(f"最大類似度: {max_similarity:.2f}")

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
import cv2
import numpy as np
import time
import os
import logging
import glob
from ultralytics import YOLO

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# YOLOv8モデルの読み込み
yolo_model = YOLO('yolov8n.pt')

# 顔検出器の読み込み
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 128))  # HOGに適したサイズに変更
        
        # HOG特徴量の計算
        hog = cv2.HOGDescriptor()
        features = hog.compute(face_resized)
        
        return features.flatten()
    return None

def load_target_images(image_dir):
    target_features = []
    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_files:
        logging.error(f"ディレクトリ '{image_dir}' に画像が見つかりません。")
        return None

    for image_path in image_files:
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"'{image_path}' を読み込めませんでした。")
            continue
        features = extract_face_features(img)
        if features is not None:
            target_features.append(features)
            logging.info(f"'{os.path.basename(image_path)}' から顔の特徴量を抽出しました。")
        else:
            logging.warning(f"'{os.path.basename(image_path)}' から顔を検出できませんでした。")

    if not target_features:
        logging.error("いずれの画像からも顔を検出できませんでした。")
        return None

    return target_features

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def detect_specific_person(frame, target_features, threshold=0.7):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 128))  # HOGに適したサイズに変更
        hog = cv2.HOGDescriptor()
        features = hog.compute(face_resized).flatten()

        for i, target in enumerate(target_features):
            similarity = cosine_similarity(features, target)
            logging.info(f"顔 ({x}, {y}) とターゲット {i} の類似度: {similarity}")
            if similarity > threshold:
                return True, (x, y, w, h)

    return False, None

def move(x, y, rotation):
    # ロボットを動かす関数（実際の環境に合わせて実装が必要）
    movement = f"x: {x}, y: {y}, rotation: {rotation}"
    logging.info(f"ロボットの移動: {movement}")

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
    person_found = False

    logging.info("カメラからの映像取得を開始します。")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logging.error("フレームの取得に失敗しました。")
            break

        person_detected, face_location = detect_specific_person(frame, target_features)

        if person_detected:
            x, y, w, h = face_location
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Target Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 人物に近づく
            center_x = x + w/2
            if center_x < frame.shape[1]/2 - 50:
                move(0, -1, 0)  # 左に移動
            elif center_x > frame.shape[1]/2 + 50:
                move(0, 1, 0)  # 右に移動
            else:
                move(1, 0, 0)  # 前進

            # 人物が画面の60%以上を占めたら写真を撮影
            if h > frame.shape[0] * 0.6:
                take_photo(frame, output_image_dir)
                person_found = True
                break

        cv2.imshow("Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or person_found:
            logging.info("プログラムを終了します。")
            break

    cap.release()
    cv2.destroyAllWindows()

    if person_found:
        logging.info("特定の人物を検出し、写真を撮影しました。")
    else:
        logging.info("特定の人物を検出できなかったか、十分近づくことができませんでした。")

if __name__ == "__main__":
    main()
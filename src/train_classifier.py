import os
import cv2
import numpy as np
import pickle
from utils.config import YUNET_MODEL_PATH, SFACE_MODEL_PATH, DATABASE_FILE, DATASET_DIR

def train_classifier():
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset doesnt exist: {DATASET_DIR}.\nAdd images of people in dataset/<name of person>/, and run train_classifier.py to add them to the database")
        return

    # Initialize OpenCV YuNet for the face detection and SFace 128D feature extraction
    empty_config = np.array([], dtype=np.uint8)
    yunet_detector = cv2.FaceDetectorYN.create(
        "onnx", np.fromfile(YUNET_MODEL_PATH, dtype=np.uint8), empty_config, (320, 320), score_threshold=0.5, nms_threshold=0.2 # use np to read file instead, since it supports æøå
    ) # face detection
    sface_recognizer = cv2.FaceRecognizerSF.create(
        "onnx", np.fromfile(SFACE_MODEL_PATH, dtype=np.uint8), empty_config
    ) # 128d feature extraction

    face_database = {}

    print(f"Processing pictures in the dataset")
    # Loop thru each person's dataset folder
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        face_database[person_name] = []

        # Loop through each person image file for the current person
        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR) # support æøå in filenames
            if img is None:
                continue

            # Resize extremely large images
            h, w = img.shape[:2]
            max_dim = 640
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                h, w = img.shape[:2]

            yunet_detector.setInputSize((w, h))
            _, faces = yunet_detector.detect(img)

            # Cheks if YuNet detects least one face in the image
            if faces is not None and len(faces) > 0:
                aligned_face = sface_recognizer.alignCrop(img, faces[0])
                embedding = sface_recognizer.feature(aligned_face)
                face_database[person_name].append(embedding)

    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(face_database, f)

    total_embeddings = sum(len(embeds) for embeds in face_database.values())
    print(f"Saved {total_embeddings} face embeddings for {len(face_database)} person(s) to '{DATABASE_FILE}'!")

train_classifier()
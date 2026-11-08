import cv2
import numpy as np

class AntiSpoofing:

    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def predict(self, frame, face):
        crop = self.crop_face(frame, face)

        if crop is None:
            return None

        # Prepare image for minifasnet
        blob = cv2.dnn.blobFromImage(
            crop,
            scalefactor=1.0,
            size=(80, 80),
            swapRB=False
        )

        # Run model inference
        self.net.setInput(blob)
        logits = self.net.forward()[0]

        # Convert logits to probabilities
        exp = np.exp(logits - np.max(logits))
        probabilities = exp / np.sum(exp)

        # Class 1 = Real, classes 0 and 2 = Fake
        predicted_class = np.argmax(probabilities)

        label = "Real" if predicted_class == 1 else "Fake"
        real_score = float(probabilities[1])

        # result 1, result 2, result 3
        # result 1: 51 fake
        # result 2: 80 real
        # result 3: 78 fake

        return {
            "label": label,
            "real_score": real_score
        }

    def crop_face(self, frame, face):
        img_h, img_w = frame.shape[:2]

        x, y, w, h = map(int, face[:4])

        if w <= 0 or h <= 0:
            return None

        # Expand the face region
        scale = min(2.7, (img_w - 1) / w, (img_h - 1) / h)

        new_w = w * scale
        new_h = h * scale

        # Find the center of the face
        center_x = x + w / 2
        center_y = y + h / 2

        # Calculate expanded bounding box
        x1 = max(0, int(center_x - new_w / 2))
        y1 = max(0, int(center_y - new_h / 2))
        x2 = min(img_w - 1, int(center_x + new_w / 2))
        y2 = min(img_h - 1, int(center_y + new_h / 2))

        if x2 < x1 or y2 < y1:
            return None

        # crop original frame
        crop = frame[y1:y2 + 1, x1:x2 + 1]

        if crop.size == 0:
            return None

        return cv2.resize(crop, (80, 80))
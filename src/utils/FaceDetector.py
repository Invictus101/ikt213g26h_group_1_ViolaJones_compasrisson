import cv2

class FaceDetector:
    def __init__(self, yunet_detector, sface_recognizer, face_database):
        self.yunet_detector = yunet_detector
        self.sface_recognizer = sface_recognizer
        self.face_database = face_database
        pass

    def detect_faces(self, img):
        height, width = img.shape[:2]
        self.yunet_detector.setInputSize((width, height))  # dynamically match frame size of video frame
        # faces: np.ndarray (N, 15) | None. columns: x, y, w, h, 5 face-keypoints (x, y), score
        return self.yunet_detector.detect(img)[1]

    def get_persons_from_face(self, aligned_face, threshold):
        """Returns map<person_name, best score>, only ppl scoring >= threshold, sorted highest first"""
        face_feature = self.sface_recognizer.feature(aligned_face)

        person_scores = {}

        # Compare the live face features against all stored reference picture embeddings in the d8abase
        # Each person keeps the score of their best matching reference picture
        for person_name, stored_embeddings in self.face_database.items():
            best_score = 0.0
            for stored_feature in stored_embeddings:
                score = self.sface_recognizer.match(face_feature, stored_feature, cv2.FaceRecognizerSF_FR_COSINE)
                best_score = max(best_score, score)
            if best_score >= threshold:
                person_scores[person_name] = best_score

        return dict(sorted(person_scores.items(), key=lambda item: item[1], reverse=True)) # sort based on score
import cv2
import numpy as np
import time

from src.capture import Capture
from src.utils.config import YUNET_MODEL_PATH


def main():
    # if we read the models normally, it wont work on systems with æøå in the path name
    # since we are too proud of our norwegian letters and refuse to change path names,
    # we loaded the models into np arrays first and then sent them in, instead of path to the files
    empty_config = np.array([], dtype=np.uint8)
    yunet_detector = cv2.FaceDetectorYN.create(
        "onnx", np.fromfile(YUNET_MODEL_PATH, dtype=np.uint8), empty_config, (320, 320),
        score_threshold=0.65, nms_threshold=0.2 # use np to read file instead, since it supports æøå
    ) # face detection

    # Viola-Jones face detection using Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    #If this is a problem ur opencv-python is probably too new as it got removed.
    #Downgrade to 4.12.0.88
    if face_cascade.empty():
        raise RuntimeError("Could not load Haar Cascade model")

    cam = Capture(src=0)

    try:
        while True:
            frame_img = cam.read_frame(target_width=1080)
            if frame_img is None:
                break

            # YuNet face detection
            height, width = frame_img.shape[:2]
            yunet_detector.setInputSize((width, height))

            start = time.perf_counter()
            faces = yunet_detector.detect(frame_img)[1]
            yunet_time = (time.perf_counter() - start) * 1000

            # Viola-Jones face detection
            start = time.perf_counter()

            gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
            viola_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            viola_time = (time.perf_counter() - start) * 1000

            yunet_img = frame_img.copy()
            viola_img = frame_img.copy()

            if faces is not None:
                for face in faces:
                    (x0, y0, width, height) = face[:4]

                    cv2.rectangle(
                        yunet_img,
                        (int(x0), int(y0)),
                        (int(x0 + width), int(y0 + height)),
                        (0, 255, 0), 2
                    ) # green frame for YuNet

            if len(viola_faces) > 0:
                for (x0, y0, width, height) in viola_faces:
                    cv2.rectangle(
                        viola_img,
                        (x0, y0),
                        (x0 + width, y0 + height),
                        (255, 0, 0), 2
                    ) # blue frame for Viola-Jones

            yunet_count = 0 if faces is None else len(faces)
            viola_count = len(viola_faces)

            # Draws boxes with detection time and number of faces
            cv2.putText(
                yunet_img,
                f"YuNet: {yunet_count} faces | {yunet_time:.1f} ms",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0), 2
            )

            cv2.putText(
                viola_img,
                f"Viola-Jones: {viola_count} faces | {viola_time:.1f} ms",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0), 2
            )

            # Displays both side by side
            comparison = np.hstack((yunet_img, viola_img))

            # Resize displayed image without affecting detection
            display_width = 1600
            if comparison.shape[1] > display_width:
                scale = display_width / comparison.shape[1]
                comparison = cv2.resize(comparison, None, fx=scale, fy=scale)

            cv2.imshow("YuNet vs Viola-Jones", comparison)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break # Press 'q' to exit window

    finally:
        cam.close()


if __name__ == "__main__":
    main()
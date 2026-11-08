import cv2

class Capture: # camera
    def __init__(self, src = 0):
        self.camera = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera: {src}")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def read_frame(self, target_width = None ):
        success, frame = self.camera.read()
        if not success:
            print ("Failed to read frame from camera (error code 1)")
            return None

        ## Resize fram to improve face recognition fps
        if target_width and frame is not None:
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            target_height = int(target_width / aspect_ratio)
            frame = cv2.resize(frame, (target_width, target_height))

        return frame

    def close(self):
        if self.camera.isOpened():
            self.camera.release()
            cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import pandas as pd
class Skeleton_generator():
    def __init__(self,static_image_mode = True ):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode) # True - for photo (best quality), False - for video

    def generate(self, image):
        # konwersja BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            print(f"Nie wykryto postaci na obrazie")
            return None

        h, w, _ = image.shape
        data = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            x_px, y_px = int(lm.x * w), int(lm.y * h)
            data.append([id, lm.x, lm.y, lm.z, lm.visibility, x_px, y_px])

        df = pd.DataFrame(data, columns=["id", "x", "y", "z", "visibility", "x_px", "y_px"])

        # Draws skeleton
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return df
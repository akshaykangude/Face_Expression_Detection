import cv2
import mediapipe as mp
import math
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

class ExpressionAnalyzer:
    def __init__(self):
        self.lip_top = 13
        self.lip_bottom = 14
        self.mouth_left = 78
        self.mouth_right = 308
        self.left_eye_top = 159
        self.left_eye_bottom = 145
        self.right_eye_top = 386
        self.right_eye_bottom = 374
        self.left_eyebrow_inner = 65
        self.right_eyebrow_inner = 295
        self.left_eyebrow_outer = 52
        self.right_eyebrow_outer = 282
        self.mouth_sensitivity = 3.0
        self.eye_sensitivity = 2.5

        self.reference_face_height = 300  # Assumed typical face height at 30-40 cm

    def get_expression(self, landmarks, face_height):
        # Compute scaling factor relative to reference face size
        scale_factor = face_height / self.reference_face_height

        mouth_dist = math.dist(landmarks[self.lip_top], landmarks[self.lip_bottom])
        mouth_width = math.dist(landmarks[self.mouth_left], landmarks[self.mouth_right])
        left_eye_dist = math.dist(landmarks[self.left_eye_top], landmarks[self.left_eye_bottom])
        right_eye_dist = math.dist(landmarks[self.right_eye_top], landmarks[self.right_eye_bottom])
        avg_eye_dist = (left_eye_dist + right_eye_dist) / 2

        eyebrow_inner_y = (landmarks[self.left_eyebrow_inner][1] + landmarks[self.right_eyebrow_inner][1]) / 2
        eyebrow_outer_y = (landmarks[self.left_eyebrow_outer][1] + landmarks[self.right_eyebrow_outer][1]) / 2
        eyebrow_inner_lift = (eyebrow_outer_y - eyebrow_inner_y) / scale_factor
        eyebrow_inner_frown = (eyebrow_inner_y - eyebrow_outer_y) / scale_factor

        mouth_ratio = (mouth_dist * self.mouth_sensitivity) / scale_factor
        eye_ratio = (avg_eye_dist * self.eye_sensitivity) / scale_factor
        mouth_width_scaled = mouth_width / scale_factor

        print(f"scale_factor: {scale_factor:.2f}, mouth_ratio: {mouth_ratio:.2f}, mouth_width: {mouth_width_scaled:.2f}, "
              f"eye_ratio: {eye_ratio:.2f}, eyebrow_inner_lift: {eyebrow_inner_lift:.2f}, eyebrow_inner_frown: {eyebrow_inner_frown:.2f}")

        if mouth_width_scaled > 100 and mouth_ratio > 12 and eye_ratio > 10:
            return "HAPPY", (0, 255, 0) # 😊
        elif mouth_width_scaled < 68:
            return "SAD", (255, 0, 0) # 😢
        elif eyebrow_inner_frown > 4.0 :#and eye_ratio < 50:
            return "STRESSED", (128, 0, 128) # 😟
        elif eye_ratio > 45:
            return "ANGRY", (0, 0, 255) # 😠
        elif 9 <= mouth_ratio <= 11 and 8 <= eye_ratio <= 11:
            return "SURPRISED", (200, 200, 200) # 😲
        elif mouth_ratio > 10:
            return "SMILING", (0, 255, 127) # 🙂
        elif eye_ratio > 13:
            return "NEUTRAL", (0, 165, 255) # 😐
        else:
            return "NEUTRAL", (200, 200, 200) # 😐

def main():
    analyzer = ExpressionAnalyzer()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    print("INFO: Access the camera is approved, the webcam is Active")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_frame)

            raw_frame = frame.copy()
            h, w = frame.shape[:2]

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                    # Compute bounding box
                    x_coords = [pt[0] for pt in landmarks]
                    y_coords = [pt[1] for pt in landmarks]
                    x_min = max(min(x_coords) - 20, 0)
                    x_max = min(max(x_coords) + 20, w)
                    y_min = max(min(y_coords) - 20, 0)
                    y_max = min(max(y_coords) + 20, h)
                    face_height = y_max - y_min

                    expression, color = analyzer.get_expression(landmarks, face_height)

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1))

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, expression, (x_min, max(y_min - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            padding = np.zeros((10, w, 3), dtype=np.uint8)
            combined = np.vstack((raw_frame, padding, frame))

            cv2.rectangle(combined, (0, 0), (w, 25), (50, 50, 50), -1)
            cv2.putText(combined, "Raw Camera View", (10, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            cv2.rectangle(combined, (0, h + 10), (w, h + 10 + 25), (50, 50, 50), -1)
            cv2.putText(combined, "Expression Analysis", (10, h + 10 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            scale = 1000.0 / combined.shape[0]
            combined_resized = cv2.resize(combined, (int(combined.shape[1] * scale), 1000))

            cv2.imshow('Real-time Facial Expression Detection', combined_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

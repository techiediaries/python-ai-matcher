import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
import subprocess

# =====================================================================
# CLASS 1: ExpressionAnalyzer
# Responsibility: Handle AI models to find faces, hands, and calculate
# mathematical features (like how open the eyes or mouth are).
# =====================================================================
class ExpressionAnalyzer:
    """
    The ExpressionAnalyzer class acts as the 'Brain' of our project.
    It encapsulates (hides away) the complex MediaPipe machine learning logic.
    """
    
    # Class Variables (Constants): These belong to the class itself, 
    # not a specific object. They are the same for every analyzer.
    LEFT_EYE_UPPER = [159, 145, 158]
    LEFT_EYE_LOWER = [23, 27, 133]
    RIGHT_EYE_UPPER = [386, 374, 385]
    RIGHT_EYE_LOWER = [253, 257, 362]
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336]
    MOUTH_OUTER = [61, 291, 39, 181, 0, 17, 269, 405]
    MOUTH_INNER = [78, 308, 95, 88]
    NOSE_TIP = 4

    def __init__(self, frame_skip: int = 2):
        """
        The constructor method (__init__). It runs automatically when you 
        create a new ExpressionAnalyzer object.
        'self' refers to the specific object being created.
        """
        self.last_features = None  # Remembers the last frame's data to save processing power
        self.frame_counter = 0     # Keeps track of how many frames we've seen
        self.frame_skip = frame_skip # How many frames to skip before re-calculating (performance boost)

        # Download or load the required AI models
        self.face_model_path = self._download_model(
            "face_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        self.hand_model_path = self._download_model(
            "hand_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

        # Initialize MediaPipe objects. We need separate ones for video (webcam) and images (memes)
        self.face_mesh_video = self._init_face_landmarker(video_mode=True)
        self.hand_detector_video = self._init_hand_landmarker(video_mode=True)
        self.face_mesh_image = self._init_face_landmarker(video_mode=False)
        self.hand_detector_image = self._init_hand_landmarker(video_mode=False)

    def _download_model(self, model_path: str, url: str) -> str:
        """
        A 'private' method (indicated by the starting underscore '_').
        This means it should only be used internally by this class, not from the outside.
        Downloads AI models if they don't already exist.
        """
        if not os.path.exists(model_path):
            print(f"Downloading {model_path}...")
            try:
                subprocess.run(['curl', '-L', url, '-o', model_path], check=True, capture_output=True)
                print(f"{model_path} downloaded successfully!")
            except subprocess.CalledProcessError:
                raise RuntimeError(f"Failed to download model. Please download manually from {url}")
        return model_path

    def _init_face_landmarker(self, video_mode: bool = True):
        """Initializes the MediaPipe Face Landmarker."""
        mode = mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.face_model_path),
            running_mode=mode,
            num_faces=1,
            min_face_detection_confidence=0.5 if video_mode else 0.3,
            min_face_presence_confidence=0.5 if video_mode else 0.3,
            min_tracking_confidence=0.5 if video_mode else 0.0
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _init_hand_landmarker(self, video_mode: bool = True):
        """Initializes the MediaPipe Hand Landmarker."""
        mode = mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=mode,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3 if video_mode else 0.0
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def extract_features(self, image: np.ndarray, is_static: bool = False) -> dict:
        """
        Public method to analyze an image and return facial/hand features as a dictionary.
        """
        # Choose the right tool depending on if it's a photo or a live video frame
        face_landmarker = self.face_mesh_image if is_static else self.face_mesh_video
        hand_landmarker = self.hand_detector_image if is_static else self.hand_detector_video

        # MediaPipe requires RGB images
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if is_static:
            face_res = face_landmarker.detect(mp_image)
            hand_res = hand_landmarker.detect(mp_image)
        else:
            self.frame_counter += 1
            # Optimization: Only process every Nth frame to keep video smooth
            if self.frame_counter % self.frame_skip != 0:
                return getattr(self, "last_features", None)
            
            face_res = face_landmarker.detect_for_video(mp_image, self.frame_counter)
            hand_res = hand_landmarker.detect_for_video(mp_image, self.frame_counter)

        # If no face is found, return nothing
        if not face_res.face_landmarks:
            return None

        landmarks = face_res.face_landmarks[0]
        # Convert landmarks to a standard NumPy array for easier math
        landmark_array = np.array([[l.x, l.y] for l in landmarks])
        
        # Calculate the actual mathematical features (eye openness, smile, etc.)
        features = self._compute_features(landmark_array, hand_res)
        self.last_features = features
        return features

    def _compute_features(self, landmark_array: np.ndarray, hand_res) -> dict:
        """
        Calculates distances between points on the face to understand the expression.
        """
        # Helper function inside a method (nested function) to calculate Eye Aspect Ratio
        def ear(upper, lower):
            vert = np.linalg.norm(landmark_array[upper] - landmark_array[lower], axis=1).mean()
            horiz = np.linalg.norm(landmark_array[upper[0]] - landmark_array[upper[-1]])
            return vert / (horiz + 1e-6) # +1e-6 prevents dividing by zero!

        left_ear = ear(self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER)
        right_ear = ear(self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER)
        avg_ear = (left_ear + right_ear) / 2.0

        # Mouth calculations
        mouth_top, mouth_bottom = landmark_array[13], landmark_array[14]
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_left, mouth_right = landmark_array[61], landmark_array[291]
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        mouth_ar = mouth_height / (mouth_width + 1e-6)
        
        inner_width = np.linalg.norm(landmark_array[78] - landmark_array[308])
        mouth_width_ratio = inner_width / (mouth_width + 1e-6)

        # Eyebrow calculations
        left_brow_y = landmark_array[self.LEFT_EYEBROW][:, 1].mean()
        right_brow_y = landmark_array[self.RIGHT_EYEBROW][:, 1].mean()
        left_eye_center = landmark_array[self.LEFT_EYE_UPPER + self.LEFT_EYE_LOWER][:, 1].mean()
        right_eye_center = landmark_array[self.RIGHT_EYE_UPPER + self.RIGHT_EYE_LOWER][:, 1].mean()
        
        left_brow_h = left_eye_center - left_brow_y
        right_brow_h = right_eye_center - right_brow_y
        avg_brow_h = (left_brow_h + right_brow_h) / 2.0

        mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2.0
        nose_tip = landmark_array[self.NOSE_TIP]
        mouth_elev = nose_tip[1] - mouth_center_y

        # Hand calculations (are hands raised?)
        num_hands = len(hand_res.hand_landmarks) if hand_res.hand_landmarks else 0
        hand_raised = 0.0
        if num_hands > 0:
            face_center = landmark_array[:, 1].mean()
            face_top = landmark_array[:, 1].min()
            wrist_y = np.array([h[0].y for h in hand_res.hand_landmarks])
            middle_y = np.array([h[12].y for h in hand_res.hand_landmarks])
            if np.any((middle_y < face_center + 0.2) | (wrist_y < face_top + 0.3)):
                hand_raised = 1.0

        # Return a dictionary combining all these mathematical values
        return {
            'eye_openness': avg_ear,
            'eyes_symmetry': abs(left_ear - right_ear),
            'mouth_openness': mouth_ar,
            'mouth_width_ratio': mouth_width_ratio,
            'mouth_elevation': mouth_elev,
            'eyebrow_height': avg_brow_h,
            'brow_symmetry': abs(left_brow_h - right_brow_h),
            'num_hands': num_hands,
            'hand_raised': hand_raised,
            'surprise_score': avg_ear * avg_brow_h * mouth_ar,
            'smile_score': mouth_width_ratio * (1.0 - mouth_ar),
            'concern_score': avg_brow_h * (1.0 - mouth_elev),
            'cheers_score': mouth_width_ratio * (1.0 - mouth_ar) * hand_raised
        }


# =====================================================================
# CLASS 2: MemeLibrary
# Responsibility: Load image files, store their features, and run the
# comparison logic to find which meme matches the user best.
# =====================================================================
class MemeLibrary:
    """
    Acts as a database for our memes. 
    It 'has-a' relationship with ExpressionAnalyzer (Dependency Injection).
    """
    CACHE_FILE = "meme_features_cache.pkl"

    def __init__(self, analyzer: ExpressionAnalyzer, assets_folder: str = "assets", meme_height: int = 480):
        # We pass the analyzer in so the library can use it to scan images
        self.analyzer = analyzer 
        self.assets_folder = assets_folder
        self.meme_height = meme_height

        self.memes = []
        self.meme_features = []

        # Weights and factors used to decide how important each feature is when matching
        self.feature_keys = [
            'surprise_score', 'smile_score', 'concern_score', 'cheers_score',
            'hand_raised', 'num_hands', 'eye_openness', 'eyes_symmetry',
            'mouth_openness', 'mouth_width_ratio', 'mouth_elevation',
            'eyebrow_height', 'brow_symmetry'
        ]
        self.feature_weights = np.array([25, 20, 20, 30, 25, 15, 20, 10, 25, 20, 15, 20, 10])
        self.feature_factors = np.array([10, 10, 10, 10, 15, 15, 5, 5, 5, 5, 5, 5, 5])

        # Automatically load the memes when the library is created
        self.load_memes()

    def load_memes(self):
        """Loads memes from disk or a cache file to save time."""
        # If we already analyzed the memes in a previous session, load the fast cache!
        if os.path.exists(self.CACHE_FILE):
            with open(self.CACHE_FILE, "rb") as f:
                self.memes, self.meme_features = pickle.load(f)
            print(f"Loaded {len(self.memes)} memes from cache.\n")
            return

        assets_path = Path(self.assets_folder)
        image_files = list(assets_path.glob("*.jpg")) + list(assets_path.glob("*.png")) + list(assets_path.glob("*.jpeg"))
        print(f"Found {len(image_files)} meme images. Extracting features... This might take a moment.")

        # Analyze multiple memes at the same time using Threads
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._process_single_meme, sorted(image_files)))

        for r in results:
            if r:
                meme, features = r
                self.memes.append(meme)
                self.meme_features.append(features)
                print(f"Loaded: {meme['name']}")

        # Save to cache so it's instant next time
        with open(self.CACHE_FILE, "wb") as f:
            pickle.dump((self.memes, self.meme_features), f)
        print(f"Total memes loaded: {len(self.memes)}\n")

    def _process_single_meme(self, img_file: Path) -> tuple:
        """Reads a single image and asks the analyzer to extract its features."""
        img = cv2.imread(str(img_file))
        if img is None:
            return None
        
        # Resize image for consistency
        h, w = img.shape[:2]
        scale = self.meme_height / h
        img_resized = cv2.resize(img, (int(w * scale), self.meme_height))
        
        # Extract features (True means it's a static image, not a video)
        features = self.analyzer.extract_features(img_resized, is_static=True)
        if features is None:
            print(f"Skipping {img_file.name}: No face detected")
            return None
            
        meme_info = {
            'image': img_resized, 
            'name': img_file.stem.replace('_', ' ').title(),
            'path': str(img_file)
        }
        return meme_info, features

    def compute_similarity(self, features1: dict, features2: dict) -> float:
        """Mathematical formula to compare two dictionaries of facial features."""
        if features1 is None or features2 is None:
            return 0.0
        
        # Convert dictionaries into NumPy arrays so we can do fast math
        vec1 = np.array([features1[k] for k in self.feature_keys])
        vec2 = np.array([features2[k] for k in self.feature_keys])
        
        diff = np.abs(vec1 - vec2)
        similarity = np.exp(-diff * self.feature_factors)
        
        # Calculate a final score based on importance weights
        return float(np.sum(self.feature_weights * similarity))

    def find_best_match(self, user_features: dict) -> tuple:
        """Compares the user's current face to all stored memes to find the winner."""
        if user_features is None or not self.memes:
            return None, 0.0
            
        # Get a similarity score for every meme in our library
        scores = np.array([self.compute_similarity(user_features, mf) for mf in self.meme_features])
        
        if len(scores) == 0:
            return None, 0.0
            
        best_idx = int(np.argmax(scores)) # Index of the highest score
        return self.memes[best_idx], scores[best_idx]


# =====================================================================
# CLASS 3: MemeMatcherApp
# Responsibility: Run the camera loop, manage the window, and glue 
# the Analyzer and Library together. 
# =====================================================================
class MemeMatcherApp:
    """
    The main Application class. 
    It initializes the other classes and contains the main while loop.
    """
    def __init__(self, assets_folder="assets"):
        # We instantiate (create objects of) our other classes here.
        self.analyzer = ExpressionAnalyzer()
        self.library = MemeLibrary(analyzer=self.analyzer, assets_folder=assets_folder)

    def run(self):
        """Starts the webcam and the main application loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\n🎥 Camera started! Press 'q' to quit\n")

        # Main Loop: Runs constantly until the user presses 'q'
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1) # Flip frame horizontally (mirror effect)

            # 1. Ask the Analyzer to look at the webcam frame
            user_features = self.analyzer.extract_features(frame)
            
            # 2. Ask the Library to find the best matching meme
            best_meme, score = self.library.find_best_match(user_features)

            # 3. Handle the User Interface (Displaying the result)
            h, w = frame.shape[:2]
            if best_meme:
                meme_img = best_meme['image']
                meme_h, meme_w = meme_img.shape[:2]
                
                # Make sure the meme matches the height of the webcam frame
                scale = h / meme_h
                new_w = int(meme_w * scale)
                meme_resized = cv2.resize(meme_img, (new_w, h))

                # Create a blank black canvas wide enough for both images side-by-side
                display = np.zeros((h, w + new_w, 3), dtype=np.uint8)
                display[:, :w] = frame               # Left side: Webcam
                display[:, w:w + new_w] = meme_resized # Right side: Meme

                # Draw UI Text boxes
                cv2.rectangle(display, (5, 5), (200, 45), (0, 0, 0), -1)
                cv2.putText(display, "YOU", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                cv2.rectangle(display, (w + 5, 5), (w + new_w - 5, 75), (0, 0, 0), -1)
                cv2.putText(display, best_meme['name'], (w + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"Match: {score:.1f}", (w + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                display = frame
                cv2.putText(display, "No face detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show the final image on screen
            cv2.imshow("Meme Matcher - Press Q to quit", display)
            
            # Check if user pressed 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Cleanup: Release camera and close windows when done
        cap.release()
        cv2.destroyAllWindows()


# =====================================================================
# PROGRAM ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    # This block only runs if you run this file directly 
    # (e.g., 'python meme_matcher.py')
    print("Meme Matcher Starting...\n")
    
    # Create the application object and run it
    app = MemeMatcherApp(assets_folder="assets")
    app.run()
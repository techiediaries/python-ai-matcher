# 🎭 Meme Matcher - Educational OOP Project

A real-time computer vision application that matches your facial expressions and hand gestures to famous internet memes using MediaPipe's face and hand detection.

**This project has been specifically designed as an educational tool to learn Object-Oriented Programming (OOP) in Python.** It features extensive comments, docstrings, and a clean, modular architecture.

## ✨ What It Does

Point your webcam at yourself, make different facial expressions and hand gestures, and watch as the app finds the meme that best matches your expression in real-time! The matched meme appears side-by-side with your camera feed.

## 🚀 Features

* **Educational OOP Architecture**: Cleanly separated classes demonstrating the Single Responsibility Principle, Encapsulation, and Dependency Injection.

* **Real-time Face Detection**: Uses MediaPipe Face Landmarker to track 478 facial landmarks.

* **Hand Gesture Detection**: Tracks hand positions to distinguish similar expressions (e.g., Leo's cheers vs Disaster Girl's smirk).

* **Advanced Expression Analysis**:

  * Eye openness (surprise, wide eyes)

  * Eyebrow position (raised, furrowed)

  * Mouth shape (smiling, open, concerned)

  * Hand gestures (raised hands, fist pumps)

* **Smart Matching Algorithm**: Weighted similarity scoring with exponential decay for accurate matching.

* **Smart Caching**: Automatically caches processed meme features so subsequent startups are instant.

## 🏗️ The OOP Architecture (How it's built)

This project is broken down into three main classes, making it easy to read, modify, and extend:

1. **`ExpressionAnalyzer` (The Brain 🧠)**

   * Encapsulates all the complex machine learning logic.

   * Responsible for initializing MediaPipe models, detecting faces/hands, and calculating the mathematical features (like eye aspect ratio or mouth width).

2. **`MemeLibrary` (The Database 📚)**

   * Responsible for loading images from the disk, using the Analyzer to extract their features, and saving them to a local cache.

   * Contains the mathematical logic to compare a user's face to the stored memes and find the best match.

3. **`MemeMatcherApp` (The Controller & UI 🖥️)**

   * The glue that holds the application together.

   * Manages the webcam, draws the OpenCV user interface, and handles the main application loop.

## 💻 Installation

### Prerequisites

* Python 3.11+

* A working webcam

### Setup

1. Clone the repository:

```bash
git clone <your-repo-link>
cd python_meme_matcher
````

2. Install dependencies:

```bash
pip install mediapipe opencv-python numpy
```


3. Run the application:

```bash
python meme_matcher.py
```


*Note: The first time you run it, the app will automatically download the required MediaPipe AI models (\~7MB total) and process your images.*

## 📸 How to Use

1. Run the app using the command above.

2. Your webcam will activate.

3. Make different expressions and gestures to trigger specific memes:

   * **Angry face** → Angry Baby

   * **Smirk (no hands)** → Disaster Girl

   * **Smirk + hand on chin** → Gene Wilder

   * **Smile + raised hand** → Leonardo DiCaprio

   * **Wide eyes/staring** → Overly Attached Girlfriend

   * **Happy + fist pump** → Success Kid

4. Press **'q'** on your keyboard to quit the application.

## 🔬 How It Works Under the Hood

### 1. Face & Hand Detection

* Uses MediaPipe Face Landmarker (478 landmarks per face) and Hand Landmarker (21 landmarks per hand, up to 2 hands).

* Detects facial features and hand positions in real-time.

### 2. Feature Extraction

For each frame, the `ExpressionAnalyzer` calculates:

* **Eye features**: Openness, symmetry

* **Eyebrow features**: Height, position relative to eyes

* **Mouth features**: Openness, width ratio, elevation

* **Hand features**: Number of hands, raised/lowered position

### 3. Similarity Matching

* The `MemeLibrary` compares your live features against pre-loaded meme features.

* Uses weighted exponential decay scoring. Higher weights are given to distinctive features (e.g., cheers score: 30 points, hand_raised: 25 points).

* Finds the highest-scoring match and the `MemeMatcherApp` displays it alongside your video feed.

## 🤝 Contributing

This is an educational project, and contributions are welcome! Feel free to:

* Add more memes to the `assets/` folder (add hand gestures for better accuracy!).

* Improve the matching weights in `MemeLibrary`.

* Create a new UI class to replace the OpenCV window with a web interface or GUI library.

## 📄 License

This project is for educational purposes.

## 🙏 Credits

* **MediaPipe**: Google's ML framework for face and hand detection

* **Meme Images**: Fair use, iconic internet memes

* **OpenCV**: Open source computer vision library

* **KristelTech**: Make Me a Meme 

You can find a tutorial that explains the process step by step from this tutorial:

[Build Your Own AI Meme Matcher: A Beginner's Guide to Computer Vision with Python](https://10xdev.blog/ai-matcher-python)



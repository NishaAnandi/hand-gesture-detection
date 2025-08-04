# ✋ Hand Gesture Detection with OpenCV

A real-time hand gesture detection project using **OpenCV** and **MediaPipe**. It captures hand movements through your webcam and classifies common gestures like thumbs up, rock sign, and finger counts.

---

## 🚀 Features

- Real-time hand tracking via webcam
- Gesture classification: Thumbs up, 1–4 fingers, rock sign, etc.
- Built with OpenCV and MediaPipe
- Lightweight and fast

---

## 📁 Project Structure

HandGestureDetectionOpenCV/
├── hand_gesture_realtime.py # Main hand gesture detection app
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── demo.png / demo.gif # (Optional) Demo visuals


---

## ⚙️ Setup Instructions

### 1. 📦 Clone the repository

```bash
git clone https://github.com/your-username/HandGestureDetectionOpenCV.git
cd HandGestureDetectionOpenCV

2. 🐍 Create Virtual Environment (optional)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. 📥 Install Dependencies
pip install -r requirements.txt

4. ▶️ Run the App
python hand_gesture_realtime.py
Press q to exit the webcam feed.
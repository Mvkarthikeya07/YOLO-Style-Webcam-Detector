🎯 YOLO-Style-Webcam-Detector

A fast, clean, lightweight object detection web app built using Flask + OpenCV (NO heavy ML models).

🖼️ Screenshots
🔐 Login Page
<img width="1366" height="768" alt="2025-11-18 (9)" src="https://github.com/user-attachments/assets/b3895653-e405-44ae-8bb4-ca1779539186" />

🖥️ Detection Dashboard (Idle)
<img width="1366" height="768" alt="2025-11-18 (10)" src="https://github.com/user-attachments/assets/bba31ff7-9ecd-443e-8926-e3149d57bd57" />

🎥 Live Detection Example
<img width="1366" height="768" alt="2025-11-18 (14)" src="https://github.com/user-attachments/assets/d02d20cc-39e8-42f0-bc36-715fbfa6c4d3" />

📌 Overview

This project is a lightweight object detection web application that simulates a YOLO-like experience without using any machine learning models.
It uses OpenCV + custom rule-based logic to detect simple objects through webcam or image upload.

It includes a clean login system, a responsive UI, and real-time object detection — perfect for beginners, students, and portfolio projects.

🌟 Key Features

🔹 Lightweight Detection (NO YOLO, NO ML)

Uses OpenCV to detect:

Shapes

Contours

Edges

Colors

Basic object patterns

🔹 Webcam + Image Upload

Works directly in the browser:

Live webcam detection

Upload image detection

Instant result

🔹 Built-in Login System

Secure, easy login flow:

/ → Login page

/home → Detection dashboard

/logout → End session

🔹 Fully Responsive UI

Using HTML + Bootstrap:

login.html

index.html

🔹 Clean Flask Backend

Clear route structure

Reliable session handling

Processes camera frames / images

Returns structured JSON detection results

📁 Project Structure

YOLO-Style-Webcam-Detector
│
├── templates/
│   ├── index.html
│   └── login.html
│
├── app.py
├── requirements.txt
└── README.md


🧠 How Object Detection Works

This project does not use a YOLO model.
Instead, it uses custom OpenCV image processing:

Gaussian blurring

Canny edge detection

Contour extraction

Shape approximation

Area & aspect-ratio filtering

Bounding-box generation

Example Output
{
  "detections": [
    { "object": "rectangle", "x": 112, "y": 80, "w": 120, "h": 98 }
  ]
}


🔧 Tech Stack

Component	Technology
Backend	Python, Flask
Image Processing	OpenCV
Frontend	HTML, CSS, Bootstrap
Auth	Flask Sessions

⚙️ Installation

1️⃣ Create Virtual Environment (optional)

python -m venv venv


2️⃣ Activate it (Windows)

venv\Scripts\activate


3️⃣ Install dependencies

pip install -r requirements.txt


4️⃣ Run the app

python app.py


5️⃣ Open in browser

http://127.0.0.1:5000


🔐 Login System Overview

User enters credentials

Credentials validated

Session created

User redirected to /home

Logout clears session

Prevents unauthorized access to detection page.

🚀 Routes

Route	Method	Description
/	GET / POST	Login page
/home	GET	Detection dashboard
/detect	POST	Processes webcam / uploaded image
/logout	GET	Clears session

🧪 Limitations

⚠ Best accuracy in good lighting
⚠ Detects only simple objects (shapes, edges, basic items)
⚠ Not a real YOLO model — rule-based detection

📌 Future Enhancements

Add Tiny YOLO / Nano YOLO as optional mode

Add object tracking

Improve shape classification

Add history of detections

Add dark/light theme switch

🎓 Perfect For

✔ College mini project
✔ Resume portfolio
✔ Demonstrating Flask + OpenCV skills
✔ Lightweight detection demo

👤 Author
M V Karthikeya
YOLO-Style Webcam Detection — 2025

📜 License

This project is licensed under the MIT License.

📦 Setup (Beginner Friendly)

If you are new to Python or want an easy start:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

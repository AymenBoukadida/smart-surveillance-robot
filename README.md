# 🤖 Smart Surveillance Robot

### 🧠 Project Overview
The **Smart Surveillance Robot** is an AI-powered security system that integrates **computer vision**, **speech recognition**, and **automation** to enhance real-time surveillance.  
It detects and recognizes individuals, identifies suspicious activity, and automatically triggers alerts through **audio, email, and SMS notifications**.

---

## 🧩 System Architecture

### 🎥 1. Camera Input
- The system uses the **Raspberry Pi Camera (Picamera2)** or a standard webcam.
- Captures live video frames continuously for analysis.

### 🧠 2. Face Recognition Module
- Uses the **face_recognition** library to:
  - Encode known faces from an `images/` directory.
  - Compare real-time faces with known encodings.
- If the face is recognized:
  - The person’s name is displayed.
  - Their attendance is logged with a timestamp.
  - A welcome message is spoken.
- If the face is **unknown**:
  - An image of the intruder is saved.
  - SMS and email alerts are triggered.

### 🧾 3. Object Detection Module
- Powered by **TensorFlow Lite** for efficient on-device inference.
- Loads a pre-trained `detect.tflite` model and `labelmap.txt` labels.
- Detects specific objects (e.g., “vol” or suspicious items).
- When detected, an alarm sound is played and an alert is sent.

### 🔊 4. Audio and Voice System
- **pyttsx3** handles text-to-speech for spoken messages.
- **SpeechRecognition** is used for voice-based password verification.
- A voice challenge is presented to verify access if needed.

### 📡 5. Alerting Mechanisms
- **Email Alerts:**  
  Sends a message with an attached image and audio file to a predefined address when an unknown person is detected.
- **SMS Alerts:**  
  Uses the **Vonage API** to instantly notify the owner via text message.
- **Alarm Sound:**  
  Plays an audio warning or alarm tone locally when triggered.

### 🧾 6. Attendance Logging
- Each recognized individual is automatically recorded in an `attendance.csv` file with their name and time of detection.

### 🧵 7. Multithreading
- Threads are used for handling simultaneous tasks:
  - Face recognition
  - Voice verification
  - Directory monitoring
  - Email/SMS alerts
- This ensures smooth real-time operation without lag.

---

## 🧱 Technical Components

| Component | Description |
|------------|-------------|
| **Language** | Python 3 |
| **AI Framework** | TensorFlow Lite |
| **Libraries** | OpenCV, face_recognition, pyttsx3, SpeechRecognition, playsound, smtplib, vonage |
| **Hardware** | Raspberry Pi / Standard Webcam |
| **Data Files** | `.tflite` model, label map, attendance log, known faces |

---

## 🖥️ Workflow Summary
1. **Camera captures frame** → Face & object detection.  
2. **Known face** → Welcome + Attendance log.  
3. **Unknown face or object** → Alarm + Email + SMS.  
4. **Optional voice authentication** for access control.  
5. Continuous monitoring with threaded background tasks.

---

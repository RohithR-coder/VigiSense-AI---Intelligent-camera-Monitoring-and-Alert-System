# SCAS AI – Smart Camera Alert System

SCAS AI is a high-precision home surveillance system that uses **YOLOv26** for object detection and **InsightFace (Buffalo_L)** for state-of-the-art face recognition.

## Features

- **High-Precision Face Recognition**: Uses InsightFace 512-D embeddings to distinguish between known residents and unknown intruders.
- **Modern Object Detection**: YOLOv26 real-time detection for people, animals, and objects.
- **Smart Alerts**: 
    - 🟩 **Known Person**: Visual confirmation, no alarm.
    - 🟥 **Unknown Person**: 🚨 **Intruder Alert!** triggers a siren and mobile notification.
    - 🟦 **Objects/Animals**: Visual tracking, no alarm.
- **Mobile Notifications**: Integration with **Pushbullet** for instant alerts.
- **Automatic Recording**: Saves incident footage to `~/scas_recordings`.
- **User-Friendly UI**: Built with `CustomTkinter` for a professional dark-themed experience.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install opencv-python numpy ultralytics insightface onnxruntime-gpu pygame pillow customtkinter pushbullet.py
    ```
    *(Note: Use `onnxruntime` if you don't have an NVIDIA GPU)*

2.  **Configuration**:
    - Open `scas.py` and replace the Pushbullet API key with your own.
    - Add photos of trusted people to the `known_faces/` folder. Name the files after the person (e.g., `Keerthi.jpg`).

3.  **Run the application**:
    ```bash
    python scas.py
    ```

## Notification Troubleshooting (Pushbullet)

If you are not receiving notifications:
1.  Check your terminal/console for `[SCAS] Pushbullet` errors.
2.  Ensure your Pushbullet account is active and has not exceeded its monthly free limit (500 pushes).
3.  Verify the API key is correct.
4.  If Pushbullet continues to fail (e.g., "Pro Required" error), consider using a different API key or another service like **NTFY**.

import tkinter as tk
from tkinter import messagebox, simpledialog
import os
import cv2
import pygame
import numpy as np
from pushbullet import Pushbullet
from ultralytics import YOLO
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
import insightface
from insightface.app import FaceAnalysis
import logging

# Initialize InsightFace with Buffalo_L model (High Accuracy)
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Disable logs for clarity
logging.getLogger("insightface").setLevel(logging.ERROR)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# YOLOv26 will automatically download the .pt model file
MODEL_PATH = 'yolo26n.pt'
COCO_NAMES_PATH = os.path.join(SCRIPT_DIR, 'coco.names')
KNOWN_FACES_DIR = os.path.join(SCRIPT_DIR, 'known_faces')
ALARM_PATH = os.path.join(os.environ.get('USERPROFILE', ''), 'Downloads', '86502^alarm.wav')
RECORDINGS_DIR = os.path.join(os.environ.get('USERPROFILE', ''), 'scas_recordings')
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
for d in [RECORDINGS_DIR, KNOWN_FACES_DIR]:
    os.makedirs(d, exist_ok=True)
with open(COCO_NAMES_PATH, 'r') as f:
    COCO_CLASSES = [line.strip() for line in f.readlines()]
try:
    yolo_model = YOLO(MODEL_PATH)
    print('[SCAS] YOLOv26 loaded OK')
except Exception as e:
    print(f'[SCAS] Error loading YOLOv26: {e}')
    yolo_model = None
known_embeddings = [] # List of {'name': str, 'embedding': np.array}
model_trained = False
MATCH_THRESHOLD = 0.65 # Optimized for InsightFace cosine similarity (higher is more lenient)
def load_known_faces():
    """
    Generates and stores InsightFace embeddings for all images in known_faces/.
    """
    global known_embeddings, model_trained
    known_embeddings = []
    
    print('[SCAS] Generating high-precision embeddings...')
    
    for root, dirs, files in os.walk(KNOWN_FACES_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            name = os.path.basename(root) if os.path.abspath(root) != os.path.abspath(KNOWN_FACES_DIR) else os.path.splitext(fname)[0]
            name = ''.join([i for i in name if not i.isdigit()]).strip().rstrip('_- ')
            
            try:
                img = cv2.imread(fpath)
                faces_found = face_app.get(img)
                
                if faces_found:
                    # Store the 512-D embedding
                    embedding = faces_found[0].normed_embedding
                    known_embeddings.append({"name": name, "embedding": embedding})
                    print(f'[SCAS] Indexed: {name} ({fname})')
            except Exception as e:
                print(f'[SCAS] Skipping {fname}: {e}')

    if known_embeddings:
        model_trained = True
        print(f'[SCAS] Ready! {len(known_embeddings)} embeddings loaded.')
    else:
        print('[SCAS] WARNING: No known faces found.')
        model_trained = False

load_known_faces()
try:
    pb = Pushbullet('o.r6ahTiaZNgGtTxjx6vFaUNgqYfKky1AU')
    print('[SCAS] Pushbullet initialized OK')
except Exception as e:
    print(f'[SCAS] Pushbullet initialization error: {e}')
    pb = None
ctk.set_appearance_mode('Dark')
ctk.set_default_color_theme('blue')
scas_active = False
cap = None
out = None

class SCASApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('SCAS AI – Smart Camera Alert System')
        self.geometry('1100x650')
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.image_ref = None
        self.cleanup_complete = threading.Event()
        self.cleanup_complete.set()
        self._alert_cooldown = False
        self._face_history = []
        self.COINCIDENCE_THRESHOLD = 2 # Reduced for faster detection
        self.CONFIDENCE_THRESHOLD = 120 # Slightly looser for better detection at distance
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=210, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky='nsew')
        self.sidebar.grid_rowconfigure(7, weight=1)
        ctk.CTkLabel(self.sidebar, text='SCAS AI', font=ctk.CTkFont(size=26, weight='bold')).grid(row=0, column=0, padx=20, pady=(20, 4))
        ctk.CTkLabel(self.sidebar, text='Smart Camera Alert System', font=ctk.CTkFont(size=10), text_color='gray').grid(row=1, column=0, padx=20, pady=(0, 10))
        self.status_label = ctk.CTkLabel(self.sidebar, text='● Inactive', text_color='#ff5555', font=ctk.CTkFont(size=13, weight='bold'))
        self.status_label.grid(row=2, column=0, padx=20, pady=6)
        
        ctk.CTkLabel(self.sidebar, text='Camera Source', font=ctk.CTkFont(size=12)).grid(row=3, column=0, padx=20, pady=(10, 0))
        self.camera_option = ctk.CTkOptionMenu(self.sidebar, values=['Laptop Camera (0)', 'Phone/DroidCam (1)', 'Phone/DroidCam (2)', 'Phone/DroidCam (3)'], width=180)
        self.camera_option.set('Phone/DroidCam (1)') # Default to Phone Cam
        self.camera_option.grid(row=4, column=0, padx=20, pady=(0, 10))

        self.activate_btn = ctk.CTkButton(self.sidebar, text='▶  Activate', fg_color='#1f8a1f', hover_color='#156315', command=self.start_activation_thread)
        self.activate_btn.grid(row=5, column=0, padx=20, pady=6)
        self.deactivate_btn = ctk.CTkButton(self.sidebar, text='■  Deactivate', fg_color='#8a1f1f', hover_color='#631515', command=self.deactivate_scas, state='disabled')
        self.deactivate_btn.grid(row=6, column=0, padx=20, pady=6)
        self.known_faces_btn = ctk.CTkButton(self.sidebar, text='👤  Manage Known Faces', command=self.manage_known_faces)
        self.known_faces_btn.grid(row=7, column=0, padx=20, pady=6)
        self.recordings_btn = ctk.CTkButton(self.sidebar, text='🎞  View Recordings', command=self.view_recordings)
        self.recordings_btn.grid(row=8, column=0, padx=20, pady=6)
        legend = ctk.CTkFrame(self.sidebar, fg_color='transparent')
        legend.grid(row=8, column=0, padx=10, pady=(10, 20), sticky='s')
        ctk.CTkLabel(legend, text='Detection Legend', font=ctk.CTkFont(size=11, weight='bold'), text_color='gray').pack(anchor='w')
        ctk.CTkLabel(legend, text='🟩  Known Person – no alert', font=ctk.CTkFont(size=10), text_color='#55ff55').pack(anchor='w')
        ctk.CTkLabel(legend, text='🟥  Unknown Person – ALERT', font=ctk.CTkFont(size=10), text_color='#ff5555').pack(anchor='w')
        ctk.CTkLabel(legend, text='🟦  Object/Animal – no alert', font=ctk.CTkFont(size=10), text_color='#55aaff').pack(anchor='w')
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky='nsew')
        self.video_label = tk.Label(self.main_frame, text='Primary Feed (Inactive)', bg='#1a1a1a', fg='#888888', font=('Helvetica', 14))
        self.video_label.pack(expand=True, fill='both')
        self.alert_bar = ctk.CTkLabel(self, text='', fg_color='transparent', font=ctk.CTkFont(size=13, weight='bold'), text_color='#ff5555')
        self.alert_bar.grid(row=1, column=0, columnspan=2, sticky='ew', padx=20, pady=(0, 6))
        self.alarm_sound = None
        try:
            pygame.mixer.init()
            if os.path.exists(ALARM_PATH):
                self.alarm_sound = pygame.mixer.Sound(ALARM_PATH)
        except Exception:
            pass
        self.update_gui_loop()

    def update_gui_loop(self):
        try:
            if scas_active:
                with self.frame_lock:
                    frame_to_show = self.current_frame.copy() if self.current_frame is not None else None
                if frame_to_show is not None:
                    frame_to_show = cv2.resize(frame_to_show, (780, 520))
                    img_pil = Image.fromarray(cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB))
                    self.image_ref = ImageTk.PhotoImage(image=img_pil)
                    self.video_label.configure(image=self.image_ref, text='')
        except Exception as e:
            print(f'[GUI] {e}')
        if self.winfo_exists():
            self.after(30, self.update_gui_loop)

    def start_activation_thread(self):
        if not self.cleanup_complete.is_set():
            messagebox.showinfo('Please Wait', 'System is still shutting down. Try again in a moment.')
            return
        self.activate_btn.configure(state='disabled')
        self.status_label.configure(text='● Initializing…', text_color='#ffaa00')
        self.cleanup_complete.clear()
        self.after(500, lambda: threading.Thread(target=self.activate_scas, daemon=True).start())

    def activate_scas(self):
        global scas_active, cap, out
        try:
            # Parse index from selection string (e.g., "Laptop Camera (0)" -> 0)
            selection = self.camera_option.get()
            import re
            match = re.search(r'\((\d+)\)', selection)
            target_idx = int(match.group(1)) if match else 1
            
            print(f'[SCAS] Activating Camera Index: {target_idx}...')
            
            # Use CAP_DSHOW for better compatibility on Windows
            cap = cv2.VideoCapture(target_idx, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(target_idx)
            
            if not cap or not cap.isOpened():
                self.after(0, lambda: messagebox.showerror('Error', f'Failed to connect to index {target_idx}. Ensure camera is connected.'))
                self.after(0, self.reset_ui)
                self.cleanup_complete.set()
                return
            
            print(f'[SCAS] Successfully connected to Index: {target_idx}')
            ts = int(pygame.time.get_ticks())
            record_file = os.path.join(RECORDINGS_DIR, f'scas_{ts}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(record_file, fourcc, 20.0, (640, 480))
            scas_active = True
            self.after(0, lambda: self.deactivate_btn.configure(state='normal'))
            self.after(0, lambda: self.status_label.configure(text='● Active', text_color='#55ff55'))
            self.detection_loop()
        except Exception as e:
            print(f'[SCAS] Activation error: {e}')
            self.after(0, self.reset_ui)
            self.cleanup_complete.set()

    def detection_loop(self):
        global scas_active, cap, out
        while scas_active:
            if cap is None:
                break
            ret, frame = cap.read()
            if not ret:
                break
            annotated = self.run_detection(frame)
            if out:
                out.write(cv2.resize(annotated, (640, 480)))
            with self.frame_lock:
                self.current_frame = annotated.copy()
        self.cleanup()

    def run_detection(self, frame):
        if yolo_model is None:
            return frame
        
        # Run inference with YOLOv26
        results = yolo_model(frame, conf=0.4, verbose=False)
        
        unknown_detected = False
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                conf = float(box.conf[0])
                cid = int(box.cls[0])
                
                label_name = result.names[cid]
                
                if label_name == 'person':
                    recognized, person_name = self.identify_face(frame, x, y, w, h)
                    if recognized:
                        color = (50, 205, 50)
                        text = f'{person_name} ({conf:.0%})'
                    else:
                        color = (50, 50, 220)
                        text = f'Unknown  ALERT ({conf:.0%})'
                        unknown_detected = True
                else:
                    color = (210, 140, 30)
                    text = f'{label_name} ({conf:.0%})'
                
                # Draw box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x, y - th - 10), (x + tw + 4, y), color, -1)
                cv2.putText(frame, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
        if unknown_detected:
            self._trigger_alert()
        else:
            self._clear_alert()
        return frame

    def identify_face(self, frame, x, y, bw, bh):
        if not model_trained:
            return (False, '')
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + bw + pad)
        y2 = min(frame.shape[0], y + bh + pad)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return (False, '')
            
        try:
            # InsightFace works best on the full ROI or even slightly padded image
            faces_found = face_app.get(roi)
            
            if not faces_found:
                self._face_history.append('NoFace')
                return (False, '')

            # Use the most prominent face in the ROI
            current_embedding = faces_found[0].normed_embedding
            best_match = None
            max_sim = -1.0

            for known in known_embeddings:
                # InsightFace normed_embeddings use dot product for cosine similarity
                similarity = np.dot(current_embedding, known["embedding"])
                
                if similarity > max_sim:
                    max_sim = similarity
                    if similarity > MATCH_THRESHOLD:
                        best_match = known["name"]

            if best_match:
                print(f'[SCAS] Recognized: {best_match} (Sim: {max_sim:.3f})')
                self._face_history.append(best_match)
            else:
                self._face_history.append('Unknown')

        except Exception as e:
            print(f'[SCAS] Detection error: {e}')
            self._face_history.append('NoFace')
            return (False, '')
            
        # Keep only the last N frames of history (smooths out missed frames)
        if len(self._face_history) > 20: # store up to 20 frames
            self._face_history.pop(0)
            
        # Check if we recognized this known person reasonably often recently
        # E.g. 3 times in the last 15 frames
        recent_history = self._face_history[-15:]
        known_counts = {}
        for f in recent_history:
            if f not in ('Unknown', 'NoFace'):
                known_counts[f] = known_counts.get(f, 0) + 1
                
        best_known = None
        max_seen = 0
        for name, count in known_counts.items():
            if count >= self.COINCIDENCE_THRESHOLD and count > max_seen:
                best_known = name
                max_seen = count
                
        if best_known:
            return (True, best_known)
            
        return (False, '')

    def _trigger_alert(self):
        if not self._alert_cooldown:
            self._alert_cooldown = True
            self.after(0, lambda: self.alert_bar.configure(text='🚨  UNKNOWN PERSON DETECTED – INTRUDER ALERT!', fg_color='#6e1111'))
            if self.alarm_sound and (not pygame.mixer.get_busy()):
                self.alarm_sound.play(-1)
            threading.Thread(target=self._send_push, daemon=True).start()
            self.after(10000, self._reset_cooldown)

    def _reset_cooldown(self):
        self._alert_cooldown = False

    def _clear_alert(self):
        # We don't reset _alert_cooldown here, so the 10s timer from _trigger_alert persists
        if pygame.mixer.get_busy():
            pygame.mixer.stop()
        self.after(0, lambda: self.alert_bar.configure(text='', fg_color='transparent'))

    def _send_push(self):
        if pb:
            try:
                print('[SCAS] Sending Pushbullet notification...')
                # Primary attempt: Send to all devices
                pb.push_note('🚨 SCAS ALERT', 'Unknown person detected by your camera!')
                print('[SCAS] Notification sent successfully.')
            except Exception as e:
                print(f'[SCAS] Pushbullet global send error: {e}')
                # Fallback: Try sending to a specific device if global fails (sometimes avoids Pro limits)
                try:
                    devices = pb.devices
                    if devices:
                        print(f'[SCAS] Fallback: Sending to device "{devices[0].nickname}"...')
                        devices[0].push_note('🚨 SCAS ALERT', 'Unknown person detected by your camera!')
                        print('[SCAS] Fallback notification sent successfully.')
                    else:
                        print('[SCAS] No devices found for fallback.')
                except Exception as e2:
                    print(f'[SCAS] Pushbullet fallback also failed: {e2}')

    def deactivate_scas(self):
        global scas_active
        scas_active = False
        self.deactivate_btn.configure(state='disabled')
        self.status_label.configure(text='● Disconnecting…', text_color='#ffaa00')

    def cleanup(self):
        global cap, out
        try:
            if cap:
                cap.release()
                cap = None
            if out:
                out.release()
                out = None
            if pygame.mixer.get_busy():
                pygame.mixer.stop()
        except Exception as e:
            print(f'[SCAS] Cleanup error: {e}')
        finally:
            self.after(500, self.finish_cleanup)

    def finish_cleanup(self):
        self.reset_ui()
        self.cleanup_complete.set()

    def reset_ui(self):
        self.activate_btn.configure(state='normal')
        self.deactivate_btn.configure(state='disabled')
        self.status_label.configure(text='● Inactive', text_color='#ff5555')
        self.alert_bar.configure(text='', fg_color='transparent')
        self.video_label.configure(image='', text='Primary Feed (Inactive)')
        with self.frame_lock:
            self.current_frame = None
            self.image_ref = None

    def manage_known_faces(self):
        os.startfile(KNOWN_FACES_DIR)
        messagebox.showinfo('Known Faces Folder', f'Add photos of trusted people here:\n{KNOWN_FACES_DIR}\n\n• Name each file after the person  (e.g. Keerthi.jpg)\n• Restart SCAS after adding new photos to retrain the model.\n\nSupported formats: .jpg  .jpeg  .png')

    def view_recordings(self):
        os.startfile(RECORDINGS_DIR)
if __name__ == '__main__':
    app = SCASApp()
    app.mainloop()
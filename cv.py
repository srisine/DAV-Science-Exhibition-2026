import queue
import threading
import traceback
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

from ui import showUIWin

WINDOW = "EcoVision AI: Webcam Feed"


def fit_to_screen(frame, screen_width, screen_height):
    height, width = frame.shape[:2]
    scale = max(screen_width / width, screen_height / height)
    resized = cv2.resize(
        frame,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_height, resized_width = resized.shape[:2]
    x0 = max(0, (resized_width - screen_width) // 2)
    y0 = max(0, (resized_height - screen_height) // 2)
    return resized[y0:y0 + screen_height, x0:x0 + screen_width]


def main():
    print("Loading AI models...")
    try:
        general_model = YOLO("yolo26l.pt")
        cloth_model = YOLO("df2.pt")
    except Exception as exc:
        print(f"Could not load AI models: {exc}")
        return

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    root = tk.Tk()
    root.title(WINDOW)
    root.geometry("1000x700")
    root.minsize(640, 420)
    root.configure(bg="#09090b")

    video_label = tk.Label(root, bg="#09090b", fg="#fafafa", text="Starting webcam...", font=("Segoe UI", 12))
    video_label.pack(fill="both", expand=True)

    status_var = tk.StringVar(value="Ready · SPACE: detect   F: fullscreen   Q: quit")
    status_label = tk.Label(
        root,
        textvariable=status_var,
        anchor="w",
        bg="#18181b",
        fg="#d4d4d8",
        padx=12,
        pady=8,
        font=("Segoe UI", 9),
    )
    status_label.pack(fill="x")

    events = queue.Queue()
    state = {"running": True, "processing": False, "fullscreen": False, "frame": None}
    valid_classes = [
        class_id
        for class_id, name in general_model.names.items()
        if name.lower() != "person"
    ]

    def close_app():
        if not state["running"]:
            return
        state["running"] = False
        try:
            camera.release()
        finally:
            root.destroy()

    def toggle_fullscreen(event=None):
        state["fullscreen"] = not state["fullscreen"]
        root.attributes("-fullscreen", state["fullscreen"])
        return "break"

    def handle_key(event):
        key = event.keysym.lower()
        if key == "q" or key == "escape":
            close_app()
        elif key == "f":
            toggle_fullscreen()
        elif key == "space":
            start_detection()
        return "break"

    def start_detection():
        if state["processing"]:
            status_var.set("Detection is already running...")
            return
        if state["frame"] is None:
            status_var.set("Waiting for a webcam frame...")
            return

        state["processing"] = True
        captured_frame = state["frame"].copy()
        status_var.set("Detecting objects... please wait")

        def detection_worker(frame):
            try:
                general_results = general_model(frame, conf=0.7, classes=valid_classes, verbose=False)
                cloth_results = cloth_model(frame, conf=0.7, verbose=False)
                annotated = general_results[0].plot()
                annotated = cloth_results[0].plot(img=annotated)
                events.put(("detection", frame, annotated, [
                    (general_results[0], general_model),
                    (cloth_results[0], cloth_model),
                ]))
            except Exception:
                events.put(("error", traceback.format_exc()))
            finally:
                events.put(("finished",))

        threading.Thread(target=detection_worker, args=(captured_frame,), daemon=True).start()

    root.bind("<KeyPress>", handle_key)
    root.protocol("WM_DELETE_WINDOW", close_app)

    def update_frame():
        if not state["running"]:
            return

        success, frame = camera.read()
        if success:
            state["frame"] = frame
            display_frame = frame
            if state["fullscreen"]:
                display_frame = fit_to_screen(
                    frame,
                    max(1, root.winfo_width()),
                    max(1, root.winfo_height() - status_label.winfo_height()),
                )
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((max(1, video_label.winfo_width()), max(1, video_label.winfo_height())))
            photo = ImageTk.PhotoImage(image=image)
            video_label.configure(image=photo, text="")
            video_label.image = photo
        else:
            status_var.set("Could not read from webcam")

        try:
            while True:
                event = events.get_nowait()
                if event[0] == "detection":
                    _, frame, annotated, result_pairs = event
                    detection_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    detection_image = Image.fromarray(detection_rgb)
                    detection_photo = ImageTk.PhotoImage(detection_image)
                    showUIWin(frame, result_pairs, annotated_image=detection_photo)
                    status_var.set("Detection complete · SPACE: detect again")
                elif event[0] == "error":
                    print(event[1])
                    status_var.set("Detection failed; see terminal for details")
                elif event[0] == "finished":
                    state["processing"] = False
        except queue.Empty:
            pass

        if state["running"]:
            root.after(15, update_frame)

    root.after(0, update_frame)
    print("System ready. Focus the webcam window and press SPACE to detect, F for fullscreen, Q to quit.")
    root.mainloop()

    state["running"] = False
    camera.release()


if __name__ == "__main__":
    main()

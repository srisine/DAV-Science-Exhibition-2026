import traceback
import cv2
from ultralytics import YOLO
import tkinter as tk
from ui import showUIWin

WINDOW = "Reduce, Reuse and Revive: Webcam Feed"


def fit_to_screen(frame, sw, sh):
    """Scale + center-crop the frame so it fills the whole screen (no gray/black bars)."""
    h, w = frame.shape[:2]
    scale = max(sw / w, sh / h)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    rh, rw = resized.shape[:2]
    x0, y0 = (rw - sw) // 2, (rh - sh) // 2
    return resized[y0:y0 + sh, x0:x0 + sw]


def main():
    print("Loading model...")
    model = YOLO("yolo26l.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open the webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    root = tk.Tk()
    root.withdraw()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, int(sw * 0.7), int(sh * 0.7))
    fullscreen = False

    valid_classes = [cls_id for cls_id, name in model.names.items() if name.lower() != 'person']

    print("Ready!  SPACE = capture | F = fullscreen on/off | Q = quit")
    print("(Click on the camera window first so it receives your key presses.)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: could not read a frame from the webcam")
            break

        view = fit_to_screen(frame, sw, sh) if fullscreen else frame
        cv2.imshow(WINDOW, view)

        try:
            root.update()
        except tk.TclError:
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            )

        elif key == 32:   
            print("SPACE pressed - running detection...")
            try:
                results = model(frame, conf=0.7, classes=valid_classes)
                count = len(results[0].boxes)
                print(f"Detected {count} object(s)")

                
                annotated = results[0].plot()
                cv2.imshow(WINDOW, fit_to_screen(annotated, sw, sh) if fullscreen else annotated)
                cv2.waitKey(1)

                showUIWin(frame, results, model)
            except Exception:
                traceback.print_exc() 
    cap.release()
    cv2.destroyAllWindows()
    try:
        root.destroy()
    except tk.TclError:
        pass


if __name__ == "__main__":
    main()
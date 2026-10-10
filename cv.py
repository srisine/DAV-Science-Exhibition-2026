import cv2
from ultralytics import YOLO
import tkinter as tk
from ui import showUIWin

def main():
    model = YOLO("yolo26l.pt")
    cap = cv2.VideoCapture(0)

    root = tk.Tk()
    root.withdraw()

    valid_classes = [cls_id for cls_id, name in model.names.items() if name.lower() != 'person']

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error")
            break

        cv2.imshow("Reduce, Reuse and Revive: Webcam Feed", frame)
        
        root.update()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:
            results = model(frame, conf=0.7, classes=valid_classes)
            annotatedFrame = results[0].plot()
            cv2.imshow("Captured Frame Detections", annotatedFrame)
            showUIWin(frame, results, model)

    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

if __name__ == "__main__":
    main()
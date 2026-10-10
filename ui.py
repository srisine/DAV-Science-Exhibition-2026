import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

def setup_window():
    win = tk.Toplevel()
    win.title("Detection Results & Crops")
    win.geometry("450x650")
    win.configure(bg="#1e1e1e")
    return win

def setup_scrollable_canvas(win):
    canvas = tk.Canvas(win, bg="#1e1e1e", highlightthickness=0)
    scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#1e1e1e")
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return scrollable_frame

def process_crop(frame, box):
    xyxy = box.xyxy[0].cpu().numpy()
    xmin, ymin, xmax, ymax = map(int, xyxy)
    h, w, _ = frame.shape
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w, xmax), min(h, ymax)
    if xmin >= xmax or ymin >= ymax:
        return None
    cropped = frame[ymin:ymax, xmin:xmax]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cropped_rgb)
    base_height = 70
    if pil_img.size[1] > 0:
        w_percent = base_height / float(pil_img.size[1])
        h_size = int(float(pil_img.size[0]) * float(w_percent))
        pil_img = pil_img.resize((max(1, h_size), base_height), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(pil_img)

def bind_interactions(elements, action, enter_action, leave_action):
    for el in elements:
        el.bind("<Button-1>", action)
        el.bind("<Enter>", enter_action)
        el.bind("<Leave>", leave_action)


#TERA KAAM YAHA H
def open_detail_window(clsName, conf):
    #ye clsName VARIABLE H YAHI OBJECT KA NAAM H JISKE BAAREME TU DATA COLLECT KAREGA
    detail_win = tk.Toplevel()
    detail_win.title(f"Details - {clsName.upper()}")
    detail_win.geometry("400x300")
    detail_win.configure(bg="#1e1e1e")

    title_label = tk.Label(detail_win, text=clsName.upper(), bg="#1e1e1e", fg="#ffffff", font=("Arial", 16, "bold"))
    title_label.pack(pady=20)

    text_one = tk.Label(
        detail_win, 
        #yaha DISPOSE METHODS KE JAGAH TU APNA DATA FILL KARDE
        text=f"DISPOSE METHODS", 
        bg="#1e1e1e", 
        fg="#cccccc", 
        font=("Arial", 10), 
        justify="center"
    )
    text_one.pack(pady=10)

    text_two = tk.Label(
        detail_win, 
        #yaha RECYCLE METHODS KE GAJAG APNA DATA FILL KARDE
        text="RECYCLE METHODS", 
        bg="#1e1e1e", 
        fg="#888888", 
        font=("Arial", 9, "italic"), 
        justify="center", 
        wraplength=350
    )
    text_two.pack(pady=10)

def create_card(parent, tk_img, clsName, conf):
    row_frame = tk.Frame(parent, bg="#2d2d2d", highlightbackground="#444444", highlightthickness=1, padx=8, pady=8, cursor="hand2")
    row_frame.pack(fill="x", expand=True, padx=10, pady=5)

    img_label = tk.Label(row_frame, image=tk_img, bg="#2d2d2d", cursor="hand2")
    img_label.pack(side="left", padx=5)

    text_label = tk.Label(
        row_frame, 
        text=f"• {clsName.upper()}\n   Confidence: {conf * 100:.1f}%", 
        bg="#2d2d2d", fg="#ffffff", font=("Arial", 10, "bold"), justify="left", cursor="hand2"
    )
    text_label.pack(side="left", padx=10)

    click_action = lambda e: open_detail_window(clsName, conf)

    def on_enter(e):
        row_frame.configure(bg="#3a3a3a")
        img_label.configure(bg="#3a3a3a")
        text_label.configure(bg="#3a3a3a")

    def on_leave(e):
        row_frame.configure(bg="#2d2d2d")
        img_label.configure(bg="#2d2d2d")
        text_label.configure(bg="#2d2d2d")

    elements = [row_frame, img_label, text_label]
    bind_interactions(elements, click_action, on_enter, on_leave)

def showUIWin(frame, results, model):
    win = setup_window()
    scrollable_frame = setup_scrollable_canvas(win)
    img_references = []
    found_any = False

    for result in results:
        for box in result.boxes:
            found_any = True
            clsId = int(box.cls[0])
            clsName = model.names[clsId]
            conf = float(box.conf[0])

            tk_img = process_crop(frame, box)
            if not tk_img:
                continue
            img_references.append(tk_img)
            create_card(scrollable_frame, tk_img, clsName, conf)

    if not found_any:
        no_obj_label = tk.Label(scrollable_frame, text="No objects detected.", bg="#1e1e1e", fg="#888888", font=("Arial", 11))
        no_obj_label.pack(padx=20, pady=30)

    win.img_refs = img_references
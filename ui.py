import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

from datFetch import getRecycleInfo
from datFetch import getDisposeInfo
from datFetch import getFacrInfo
from datFetch import getSort

def setup_window():
    win = tk.Toplevel()
    win.title("Detection Results")
    win.geometry("440x620")
    win.configure(bg="#191919")
    return win

def setup_scrollable_canvas(win):
    canvas = tk.Canvas(win, bg="#191919", highlightthickness=0)
    scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#191919")
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
    base_height = 65
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

def open_detail_window(clsName, conf):
    detail_win = tk.Toplevel()
    detail_win.title(f"{clsName} — Info")
    detail_win.geometry("400x440")
    detail_win.configure(bg="#191919")

    header_frame = tk.Frame(detail_win, bg="#242424", pady=12)
    header_frame.pack(fill="x")

    title_label = tk.Label(header_frame, text=clsName.upper(), bg="#242424", fg="#f0f0f0", font=("Courier", 14, "bold"))
    title_label.pack()

    sort_label = tk.Label(header_frame, text="[Loading...]", bg="#242424", fg="#4ecca3", font=("Courier", 9, "bold"))
    sort_label.pack(pady=(2, 0))

    content_frame = tk.Frame(detail_win, bg="#191919", padx=20, pady=15)
    content_frame.pack(fill="both", expand=True)

    loading_label = tk.Label(content_frame, text="Fetching insights...", bg="#191919", fg="#888888", font=("Arial", 10, "italic"))
    loading_label.pack(pady=40)

    detail_win.update_idletasks()
    detail_win.update()

    rec_res = getRecycleInfo(clsName)
    di_res = getDisposeInfo(clsName)
    fact_res = getFacrInfo(clsName)
    sort = getSort(clsName)

    if not detail_win.winfo_exists():
        return

    loading_label.destroy()
    sort_label.config(text=f"[{sort}]")

    def add_section(parent, heading, text, color):
        tk.Label(parent, text=heading, bg="#191919", fg="#888888", font=("Arial", 9, "bold")).pack(anchor="w", pady=(8, 2))
        tk.Label(parent, text=text, bg="#191919", fg=color, font=("Arial", 10), justify="left", wraplength=350).pack(anchor="w")

    add_section(content_frame, "HOW TO RECYCLE", rec_res, "#dddddd")
    add_section(content_frame, "HOW TO DISPOSE", di_res, "#aaaaaa")
    add_section(content_frame, "FUN FACT", fact_res if fact_res else "No cool facts recorded yet!", "#66b3ff")

def create_card(parent, tk_img, clsName, conf):
    row_frame = tk.Frame(parent, bg="#212121", highlightbackground="#333333", highlightthickness=1, padx=10, pady=8, cursor="hand2")
    row_frame.pack(fill="x", expand=True, padx=12, pady=5)

    img_label = tk.Label(row_frame, image=tk_img, bg="#212121", cursor="hand2")
    img_label.image = tk_img
    img_label.pack(side="left", padx=(0, 10))

    info_frame = tk.Frame(row_frame, bg="#212121", cursor="hand2")
    info_frame.pack(side="left", fill="both", expand=True)

    name_label = tk.Label(info_frame, text=clsName.upper(), bg="#212121", fg="#ffffff", font=("Courier", 11, "bold"), justify="left", anchor="w", cursor="hand2")
    name_label.pack(anchor="w")

    conf_label = tk.Label(info_frame, text=f"match: {conf * 100:.0f}%", bg="#212121", fg="#4ecca3", font=("Courier", 9), justify="left", anchor="w", cursor="hand2")
    conf_label.pack(anchor="w", pady=(2, 0))

    click_action = lambda e: open_detail_window(clsName, conf)

    def on_enter(e):
        row_frame.configure(bg="#2a2a2a", highlightbackground="#4ecca3")
        img_label.configure(bg="#2a2a2a")
        info_frame.configure(bg="#2a2a2a")
        name_label.configure(bg="#2a2a2a")
        conf_label.configure(bg="#2a2a2a")

    def on_leave(e):
        row_frame.configure(bg="#212121", highlightbackground="#333333")
        img_label.configure(bg="#212121")
        info_frame.configure(bg="#212121")
        name_label.configure(bg="#212121")
        conf_label.configure(bg="#212121")

    elements = [row_frame, img_label, info_frame, name_label, conf_label]
    bind_interactions(elements, click_action, on_enter, on_leave)

def showUIWin(frame, results, model):
    win = setup_window()
    scrollable_frame = setup_scrollable_canvas(win)
    found_any = False

    header_note = tk.Label(scrollable_frame, text="-- DETECTED ITEMS --", bg="#191919", fg="#555555", font=("Courier", 9, "bold"))
    header_note.pack(pady=(15, 5))

    for result in results:
        for box in result.boxes:
            found_any = True
            clsId = int(box.cls[0])
            clsName = model.names[clsId]
            conf = float(box.conf[0])

            tk_img = process_crop(frame, box)
            if not tk_img:
                continue
            create_card(scrollable_frame, tk_img, clsName, conf)

    if not found_any:
        no_obj_label = tk.Label(scrollable_frame, text="hmm, couldn't find anything.", bg="#191919", fg="#666666", font=("Courier", 10))
        no_obj_label.pack(padx=20, pady=40)
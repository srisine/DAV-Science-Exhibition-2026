import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageOps
import cv2

from datFetch import getRecycleInfo
from datFetch import getDisposeInfo
from datFetch import getFacrInfo
from datFetch import getSort

#Theme
BG = "#121614"
SURFACE = "#1c221f"
SURFACE_HOVER = "#262f2a"
BORDER = "#2e3833"
TEXT = "#eef3ef"
MUTED = "#8a9a91"
ACCENT = "#4ade80"

FONT = "Segoe UI"  


def conf_color(conf):
    if conf >= 0.90:
        return "#4ade80"  
    if conf >= 0.80:
        return "#a3e635"  
    return "#fbbf24"      


def category_color(sort):
    s = str(sort).lower()
    if any(k in s for k in ("hazard", "toxic", "e-waste", "ewaste", "electronic")):
        return "#f87171"  
    if any(k in s for k in ("organic", "compost", "bio", "food")):
        return "#fb923c"  
    if "reus" in s:
        return "#60a5fa"  
    if "recycl" in s:
        return "#4ade80"  
    return "#9ca3af"      


# Helpers
def setup_window():
    win = tk.Toplevel()
    win.title("Detection Results")
    win.geometry("460x700")
    win.minsize(400, 400)
    win.configure(bg=BG)

    header = tk.Frame(win, bg=BG)
    header.pack(fill="x", padx=20, pady=(18, 6))
    tk.Label(header, text="♻  Detection Results", bg=BG, fg=TEXT,
             font=(FONT, 18, "bold")).pack(anchor="w")
    win.subtitle = tk.Label(header, text="", bg=BG, fg=MUTED, font=(FONT, 10))
    win.subtitle.pack(anchor="w", pady=(2, 0))
    tk.Frame(win, bg=ACCENT, height=2).pack(fill="x", padx=20, pady=(8, 6))
    return win


def setup_scrollable_canvas(win):
    style = ttk.Style(win)
    style.theme_use("clam")
    style.configure("Dark.Vertical.TScrollbar", background=BORDER, troughcolor=BG,
                    bordercolor=BG, arrowcolor=MUTED, lightcolor=BORDER, darkcolor=BORDER)
    style.map("Dark.Vertical.TScrollbar", background=[("active", MUTED)])

    container = tk.Frame(win, bg=BG)
    container.pack(fill="both", expand=True)

    canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview,
                              style="Dark.Vertical.TScrollbar")
    frame = tk.Frame(canvas, bg=BG)

    window_id = canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def on_wheel(event):
        if canvas.yview() == (0.0, 1.0):  
            return
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")
        elif sys.platform == "darwin":
            canvas.yview_scroll(-event.delta, "units")
        else:
            canvas.yview_scroll(int(-event.delta / 120), "units")

    
    win.bind("<MouseWheel>", on_wheel)
    win.bind("<Button-4>", on_wheel)
    win.bind("<Button-5>", on_wheel)
    return frame


def crop_box(frame, box):
    xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())
    h, w, _ = frame.shape
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w, xmax), min(h, ymax)
    if xmin >= xmax or ymin >= ymax:
        return None
    cropped_rgb = cv2.cvtColor(frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)


def make_thumb(pil_img, size, radius=14):
    """Square, center-cropped thumbnail with rounded corners."""
    img = ImageOps.fit(pil_img, (size, size), Image.Resampling.LANCZOS).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    img.putalpha(mask)
    return ImageTk.PhotoImage(img)


def bind_interactions(elements, action, enter_action, leave_action):
    for el in elements:
        el.bind("<Button-1>", action)
        el.bind("<Enter>", enter_action)
        el.bind("<Leave>", leave_action)


# Detail window
def make_section(parent, icon, title, accent):
    card = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
    card.pack(fill="x", padx=16, pady=6)

    tk.Frame(card, bg=accent, width=4).pack(side="left", fill="y")

    body_frame = tk.Frame(card, bg=SURFACE)
    body_frame.pack(side="left", fill="both", expand=True, padx=14, pady=12)

    tk.Label(body_frame, text=f"{icon}  {title}", bg=SURFACE, fg=TEXT,
             font=(FONT, 11, "bold")).pack(anchor="w")

    body = tk.Label(body_frame, text="Fetching…", bg=SURFACE, fg=MUTED,
                    font=(FONT, 10, "italic"), justify="left", anchor="w", wraplength=340)
    body.pack(anchor="w", pady=(6, 0), fill="x")

    card.bind("<Configure>", lambda e: body.configure(wraplength=max(120, e.width - 60)))
    return body


def open_detail_window(clsName, conf, pil_img=None):
    win = tk.Toplevel()
    win.title(f"Details - {clsName.upper()}")
    win.geometry("460x700")
    win.minsize(400, 400)
    win.configure(bg=BG)

    # Header
    header = tk.Frame(win, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
    header.pack(fill="x", padx=16, pady=(16, 6))
    head_inner = tk.Frame(header, bg=SURFACE)
    head_inner.pack(fill="x", padx=14, pady=14)

    if pil_img is not None:
        thumb = make_thumb(pil_img, 96, 18)
        img_label = tk.Label(head_inner, image=thumb, bg=SURFACE)
        img_label.image = thumb
        img_label.pack(side="left")

    info = tk.Frame(head_inner, bg=SURFACE)
    info.pack(side="left", fill="x", expand=True, padx=(14, 0))

    tk.Label(info, text=clsName.replace("_", " ").title(), bg=SURFACE, fg=TEXT,
             font=(FONT, 17, "bold"), anchor="w").pack(anchor="w")
    tk.Label(info, text=f"Confidence  {conf * 100:.1f}%", bg=SURFACE, fg=conf_color(conf),
             font=(FONT, 10, "bold"), anchor="w").pack(anchor="w", pady=(2, 8))

    badge = tk.Label(info, text="  Analyzing…  ", bg=BORDER, fg=MUTED,
                     font=(FONT, 9, "bold"), pady=2)
    badge.pack(anchor="w")

    # Scrollable section
    content = setup_scrollable_canvas(win)
    recycle_body = make_section(content, "♻", "How to Recycle", "#4ade80")
    dispose_body = make_section(content, "🗑", "How to Dispose", "#fb923c")
    fact_body = make_section(content, "💡", "Cool Fact", "#60a5fa")
    tk.Frame(content, bg=BG, height=12).pack()

    
    q = queue.Queue()
    jobs = (
        ("sort", getSort),
        ("recycle", getRecycleInfo),
        ("dispose", getDisposeInfo),
        ("fact", getFacrInfo),
    )

    def worker():
        for key, fn in jobs:
            try:
                q.put((key, str(fn(clsName))))
            except Exception as e:
                q.put((key, f"Could not load this info ({e})"))

    threading.Thread(target=worker, daemon=True).start()

    received = {"n": 0}

    def apply(key, value):
        if key == "sort":
            color = category_color(value)
            badge.configure(text=f"  {value.upper()}  ", bg=color, fg="#0b0f0d")
            return
        label = {"recycle": recycle_body, "dispose": dispose_body, "fact": fact_body}[key]
        label.configure(text=value, fg="#cfd8d2", font=(FONT, 10))

    def poll():
        if not win.winfo_exists():
            return
        try:
            while True:
                key, value = q.get_nowait()
                received["n"] += 1
                apply(key, value)
        except queue.Empty:
            pass
        if received["n"] < len(jobs):
            win.after(100, poll)

    poll()


#Cards
def create_card(parent, pil_img, clsName, conf):
    card = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER,
                    highlightthickness=1, cursor="hand2")
    card.pack(fill="x", padx=16, pady=6)

    inner = tk.Frame(card, bg=SURFACE, cursor="hand2")
    inner.pack(fill="x", padx=12, pady=12)

    thumb = make_thumb(pil_img, 76, 14)
    img_label = tk.Label(inner, image=thumb, bg=SURFACE, cursor="hand2")
    img_label.image = thumb  # keep a reference
    img_label.pack(side="left")

    chevron = tk.Label(inner, text="›", bg=SURFACE, fg=MUTED, font=(FONT, 22), cursor="hand2")
    chevron.pack(side="right", padx=(8, 2))

    info = tk.Frame(inner, bg=SURFACE, cursor="hand2")
    info.pack(side="left", fill="x", expand=True, padx=14)

    name_label = tk.Label(info, text=clsName.replace("_", " ").title(), bg=SURFACE, fg=TEXT,
                          font=(FONT, 13, "bold"), anchor="w", cursor="hand2")
    name_label.pack(anchor="w")

    color = conf_color(conf)
    conf_label = tk.Label(info, text=f"Confidence  {conf * 100:.1f}%", bg=SURFACE, fg=color,
                          font=(FONT, 9, "bold"), anchor="w", cursor="hand2")
    conf_label.pack(anchor="w", pady=(2, 6))

    bar = tk.Canvas(info, height=6, bg=SURFACE, highlightthickness=0, cursor="hand2")
    bar.pack(fill="x")

    def draw_bar(event=None):
        bar.delete("all")
        w = bar.winfo_width()
        bar.create_rectangle(0, 0, w, 6, fill=BORDER, width=0)
        bar.create_rectangle(0, 0, int(w * conf), 6, fill=color, width=0)

    bar.bind("<Configure>", draw_bar)

    widgets = [card, inner, img_label, chevron, info, name_label, conf_label, bar]

    def on_enter(e):
        for w in widgets:
            w.configure(bg=SURFACE_HOVER)
        card.configure(highlightbackground=ACCENT)
        chevron.configure(fg=ACCENT)

    def on_leave(e):
        for w in widgets:
            w.configure(bg=SURFACE)
        card.configure(highlightbackground=BORDER)
        chevron.configure(fg=MUTED)

    click_action = lambda e: open_detail_window(clsName, conf, pil_img)
    bind_interactions(widgets, click_action, on_enter, on_leave)


#Entry point
def showUIWin(frame, results, model):
    win = setup_window()
    scrollable_frame = setup_scrollable_canvas(win)
    count = 0

    for result in results:
        for box in result.boxes:
            clsId = int(box.cls[0])
            clsName = model.names[clsId]
            conf = float(box.conf[0])

            pil_img = crop_box(frame, box)
            if pil_img is None:
                continue
            create_card(scrollable_frame, pil_img, clsName, conf)
            count += 1

    if count == 0:
        empty = tk.Frame(scrollable_frame, bg=BG)
        empty.pack(pady=70)
        tk.Label(empty, text="🔍", bg=BG, fg=MUTED, font=(FONT, 36)).pack()
        tk.Label(empty, text="No objects detected", bg=BG, fg=TEXT,
                 font=(FONT, 13, "bold")).pack(pady=(8, 2))
        tk.Label(empty, text="Try better lighting or move the item closer.", bg=BG, fg=MUTED,
                 font=(FONT, 10)).pack()
        win.subtitle.configure(text="Nothing found in this capture")
    else:
        word = "item" if count == 1 else "items"
        win.subtitle.configure(text=f"{count} {word} detected · click a card for details")
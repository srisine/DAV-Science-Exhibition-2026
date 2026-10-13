import json
import queue
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate

import cv2
from PIL import Image, ImageTk
import ollama


BG = "#09090b"
PANEL = "#18181b"
PANEL_HOVER = "#27272a"
FG = "#fafafa"
MUTED = "#a1a1aa"
GREEN = "#34d399"


def setup_window(title, geometry, minimum):
    win = tk.Toplevel()
    win.title(title)
    win.geometry(geometry)
    win.minsize(*minimum)
    win.configure(bg=BG)
    return win


def setup_scrollable_canvas(win):
    container = tk.Frame(win, bg=BG)
    container.pack(fill="both", expand=True)

    canvas = tk.Canvas(container, bg=BG, highlightthickness=0, bd=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=BG)

    scrollable_frame.bind(
        "<Configure>",
        lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
    )

    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.bind(
        "<Configure>",
        lambda event: canvas.itemconfigure(canvas_window, width=event.width),
    )
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def on_mousewheel(event):
        if getattr(event, "delta", 0):
            canvas.yview_scroll(int(-event.delta / 120), "units")
        elif getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")

    canvas.bind("<MouseWheel>", on_mousewheel)
    canvas.bind("<Button-4>", on_mousewheel)
    canvas.bind("<Button-5>", on_mousewheel)
    return scrollable_frame


def process_crop(frame, box):
    try:
        xyxy = box.xyxy[0].detach().cpu().numpy()
        xmin, ymin, xmax, ymax = map(int, xyxy)
        height, width = frame.shape[:2]
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(width, xmax), min(height, ymax)
        if xmin >= xmax or ymin >= ymax:
            return None

        crop = frame[ymin:ymax, xmin:xmax]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb)
        image.thumbnail((120, 70), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)
    except Exception:
        return None


def open_chat_window(class_name, cached_data):
    chat_win = setup_window(
        f"{class_name.upper()} — Ecovision Chat",
        "480x600",
        (380, 480),
    )

    header = tk.Frame(chat_win, bg=PANEL, padx=14, pady=14)
    header.pack(fill="x", padx=12, pady=12)
    
    title_row = tk.Frame(header, bg=PANEL)
    title_row.pack(fill="x")
    tk.Label(
        title_row,
        text=f"💬 Chat with Ecovision Assistant",
        bg=PANEL,
        fg=FG,
        font=("Segoe UI", 11, "bold"),
    ).pack(side="left")
    
    status_indicator = tk.Label(
        title_row,
        text="● Active",
        bg=PANEL,
        fg=GREEN,
        font=("Segoe UI", 9, "bold"),
    )
    status_indicator.pack(side="right")

    tk.Label(
        header,
        text=f"Ask anything about recycling or upcycling this {class_name}.",
        bg=PANEL,
        fg=MUTED,
        font=("Segoe UI", 8),
    ).pack(anchor="w", pady=(3, 0))

    chat_container = tk.Frame(chat_win, bg=BG)
    chat_container.pack(fill="both", expand=True, padx=12, pady=(0, 6))

    chat_canvas = tk.Canvas(chat_container, bg=BG, highlightthickness=0, bd=0)
    chat_scrollbar = ttk.Scrollbar(chat_container, orient="vertical", command=chat_canvas.yview)
    msg_frame = tk.Frame(chat_canvas, bg=BG)

    msg_frame.bind(
        "<Configure>",
        lambda e: chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
    )
    chat_win_window = chat_canvas.create_window((0, 0), window=msg_frame, anchor="nw")
    chat_canvas.bind(
        "<Configure>",
        lambda e: chat_canvas.itemconfigure(chat_win_window, width=e.width)
    )
    chat_canvas.configure(yscrollcommand=chat_scrollbar.set)
    chat_canvas.pack(side="left", fill="both", expand=True)
    chat_scrollbar.pack(side="right", fill="y")

    input_frame = tk.Frame(chat_win, bg=BG, padx=12, pady=12)
    input_frame.pack(fill="x", side="bottom")

    entry_var = tk.StringVar()
    result_queue = queue.Queue()
    busy = {"value": False}
    closed = {"value": False}

    audio_stream_data = {"recording": False, "frames": []}
    SAMPLE_RATE = 16000

    if not hasattr(open_chat_window, "whisper_model"):
        print("Loading local Faster Whisper model for translation...")
        open_chat_window.whisper_model = WhisperModel("medium", device="auto", compute_type="int8")

    sort_context = cached_data.get("sort", "Not specified")
    recycle_context = cached_data.get("recycle", "Not specified")
    dispose_context = cached_data.get("dispose", "Not specified")
    fact_context = cached_data.get("fact", "Not specified")

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a friendly, expert eco-friendly assistant and your name is Ecovision AI and you are helping with "
                f"upcycling, recycling, and safe disposal of {class_name}. "
                f"Here is the complete analysis previously provided to the user:\n"
                f"- Category: {sort_context}\n"
                f"- Upcycling/Recycling method: {recycle_context}\n"
                f"- Disposal method: {dispose_context}\n"
                f"- Eco Fact: {fact_context}\n"
                "You must strictly maintain consistency with these guidelines and never contradict, alter, or change them in your chat responses. Keep answers helpful, concise, and engaging."
            ),
        }
    ]

    def append_bubble(sender, message):
        bubble_wrapper = tk.Frame(msg_frame, bg=BG)
        bubble_wrapper.pack(fill="x", pady=4, padx=4)

        is_user = sender == "You"
        bg_color = "#27272a" if is_user else PANEL
        fg_color = FG if is_user else "#f4f4f5"
        align = "e" if is_user else "w"

        bubble = tk.Frame(bubble_wrapper, bg=bg_color, padx=12, pady=10)
        bubble.pack(anchor=align)

        tk.Label(
            bubble,
            text="You:" if is_user else "Ecovision Assistant:",
            bg=bg_color,
            fg=GREEN if not is_user else MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", pady=(0, 2))

        tk.Label(
            bubble,
            text=message,
            bg=bg_color,
            fg=fg_color,
            font=("Segoe UI", 10),
            justify="left",
            wraplength=340,
        ).pack(anchor="w")

        chat_canvas.update_idletasks()
        chat_canvas.yview_moveto(1.0)

    append_bubble("AI", f"Hi! What would you like to know about this {class_name}?")

    def external_translate_to_native(text, target_lang):
        if target_lang == "en" or not target_lang:
            return text
        try:
            installed_packages = argostranslate.package.get_installed_packages()
            package_to_use = next(
                filter(lambda x: x.from_code == "en" and x.to_code == target_lang, installed_packages),
                None
            )
            
            if package_to_use:
                return argostranslate.translate.translate(text, "en", target_lang)
            else:
                return text
        except Exception:
            return text
    def worker(query, history, detected_lang=None):
        try:
            response = ollama.chat(model="llama3.1:8b", messages=history)
            reply = response["message"]["content"].strip()
            if detected_lang and detected_lang != "en":
                reply = external_translate_to_native(reply, detected_lang)
            result_queue.put(("AI", reply, query))
        except Exception:
            result_queue.put(("System", "Could not connect to AI assistant right now.", query))

    def process_and_send(query, detected_lang=None):
        if not query or busy["value"]:
            return
        
        append_bubble("You", query)
        history = messages.copy() + [{"role": "user", "content": query}]
        busy["value"] = True
        send_btn.configure(state="disabled", text="...")
        voice_btn.configure(state="disabled")
        status_indicator.configure(text="Thinking...", fg="#facc15")

        threading.Thread(target=worker, args=(query, history, detected_lang), daemon=True).start()

    def send_message(event=None):
        query = entry_var.get().strip()
        entry_var.set("")
        process_and_send(query)
        return "break"

    def audio_callback(indata, frames, time_info, status):
        if audio_stream_data["recording"]:
            audio_stream_data["frames"].append(indata.copy())

    def start_recording(event=None):
        if busy["value"]:
            return
        audio_stream_data["recording"] = True
        audio_stream_data["frames"] = []
        status_indicator.configure(text="Listening...", fg="#ef4444")
        voice_btn.configure(bg="#dc2626")

    def stop_recording_and_translate(event=None):
        if not audio_stream_data["recording"]:
            return
        audio_stream_data["recording"] = False
        status_indicator.configure(text="Translating...", fg="#3b82f6")
        voice_btn.configure(bg="#6366f1")

        def translate_worker():
            try:
                if not audio_stream_data["frames"]:
                    status_indicator.configure(text="● Active", fg=GREEN)
                    voice_btn.configure(bg="#6366f1")
                    return

                audio_flat = np.concatenate(audio_stream_data["frames"], axis=0).flatten()
                
                if len(audio_flat) < SAMPLE_RATE * 0.5: 
                    result_queue.put(("System", "Audio too short, please try again.", ""))
                    return

                segments, info = open_chat_window.whisper_model.transcribe(
                    audio_flat, beam_size=5, vad_filter=True, task="translate"
                )
                
                translated_text = " ".join([segment.text for segment in segments]).strip()
                detected_lang = info.language

                if translated_text:
                    result_queue.put(("VoiceQuery", translated_text, detected_lang))
                else:
                    result_queue.put(("System", "No speech detected in audio.", ""))

            except Exception as e:
                result_queue.put(("System", f"Translation error: {str(e)}", ""))
            finally:
                status_indicator.configure(text="Active", fg=GREEN)
                voice_btn.configure(bg="#6366f1")

        threading.Thread(target=translate_worker, daemon=True).start()

    mic_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
    mic_stream.start()

    def poll_results():
        if closed["value"]:
            return
        try:
            while True:
                item = result_queue.get_nowait()
                sender = item[0]
                if sender == "AI":
                    reply, query = item[1], item[2]
                    messages.append({"role": "user", "content": query})
                    messages.append({"role": "assistant", "content": reply})
                    append_bubble("AI", reply)
                    busy["value"] = False
                    send_btn.configure(state="normal", text="Send")
                    voice_btn.configure(state="normal")
                elif sender == "VoiceQuery":
                    query, detected_lang = item[1], item[2]
                    process_and_send(query, detected_lang)
                elif sender == "System":
                    reply = item[1]
                    append_bubble("System", reply)
                    busy["value"] = False
                    send_btn.configure(state="normal", text="Send")
                    voice_btn.configure(state="normal")
        except queue.Empty:
            pass
        if chat_win.winfo_exists():
            chat_win.after(100, poll_results)

    def on_close():
        closed["value"] = True
        try:
            mic_stream.stop()
            mic_stream.close()
        except:
            pass
        chat_win.destroy()

    entry_box = tk.Entry(
        input_frame,
        textvariable=entry_var,
        bg=PANEL,
        fg=FG,
        insertbackground=FG,
        font=("Segoe UI", 10),
        bd=0,
        highlightthickness=1,
        highlightbackground="#27272a",
    )
    entry_box.pack(side="left", fill="x", expand=True, ipady=8, ipadx=8, padx=(0, 8))
    entry_box.bind("<Return>", send_message)

    voice_btn = tk.Button(
        input_frame,
        text="🎤",
        bg="#6366f1",
        fg="#ffffff",
        font=("Segoe UI", 10, "bold"),
        bd=0,
        activebackground="#4f46e5",
        activeforeground="#ffffff",
        padx=10,
        pady=6,
        cursor="hand2",
    )
    voice_btn.pack(side="right", padx=(0, 6))
    
    voice_btn.bind("<ButtonPress-1>", start_recording)
    voice_btn.bind("<ButtonRelease-1>", stop_recording_and_translate)

    send_btn = tk.Button(
        input_frame,
        text="Send",
        command=send_message,
        bg="#10b981",
        fg="#ffffff",
        font=("Segoe UI", 9, "bold"),
        bd=0,
        activebackground="#059669",
        activeforeground="#ffffff",
        padx=14,
        pady=6,
        cursor="hand2",
    )
    send_btn.pack(side="right")
    
    chat_win.protocol("WM_DELETE_WINDOW", on_close)
    chat_win.after(100, poll_results)
    entry_box.focus_set()

def open_detail_window(class_name, confidence):
    from datFetch import getRecycleInfo, getDisposeInfo, getFacrInfo, getSort

    detail_win = setup_window(
        f"{class_name.upper()} — Details",
        "460x640",
        (400, 500),
    )
    scrollable_frame = setup_scrollable_canvas(detail_win)

    header = tk.Frame(scrollable_frame, bg=PANEL, pady=16, padx=18)
    header.pack(fill="x", pady=(14, 12), padx=14)
    tk.Label(
        header,
        text=class_name.upper(),
        bg=PANEL,
        fg=FG,
        font=("Segoe UI", 15, "bold"),
    ).pack(anchor="w")
    tk.Label(
        header,
        text=f"Detection confidence: {confidence * 100:.1f}%",
        bg=PANEL,
        fg=MUTED,
        font=("Segoe UI", 9),
    ).pack(anchor="w", pady=(3, 6))
    sort_label = tk.Label(
        header,
        text="CATEGORY: LOADING...",
        bg=PANEL,
        fg="#facc15",
        font=("Segoe UI", 9, "bold"),
    )
    sort_label.pack(anchor="w", pady=(4, 0))

    def create_section(parent, heading):
        section = tk.Frame(parent, bg=PANEL, padx=16, pady=14)
        section.pack(fill="x", pady=(0, 10), padx=14)
        tk.Label(
            section,
            text=heading,
            bg=PANEL,
            fg="#a1a1aa",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        value = tk.Label(
            section,
            text="Loading...",
            bg=PANEL,
            fg="#d4d4d8",
            font=("Segoe UI", 10),
            justify="left",
            wraplength=370,
        )
        value.pack(anchor="w", pady=(6, 0))
        return value

    recycle_label = create_section(scrollable_frame, "HOW TO UPCYCLE")
    dispose_label = create_section(scrollable_frame, "SAFE DISPOSAL")
    fact_label = create_section(scrollable_frame, "ECO FACT")

    cached_data = {"sort": "", "recycle": "", "dispose": "", "fact": ""}
    loaded_keys = set()

    chat_button = tk.Button(
        scrollable_frame,
        text="Open Ecovision AI Assistant",
        command=lambda: open_chat_window(class_name, cached_data),
        bg=PANEL,
        fg=GREEN,
        font=("Segoe UI", 10, "bold"),
        bd=0,
        activebackground=PANEL_HOVER,
        activeforeground=GREEN,
        padx=16,
        pady=12,
        cursor="hand2",
        highlightbackground=GREEN,
        highlightthickness=1,
    )

    results = queue.Queue()
    closed = {"value": False}

    def fetch_data():
        functions = (
            ("sort", getSort),
            ("recycle", getRecycleInfo),
            ("dispose", getDisposeInfo),
            ("fact", getFacrInfo),
        )
        for key, function in functions:
            try:
                value = function(class_name)
            except Exception:
                value = "Could not load information right now."
            results.put((key, value))

    def poll_results():
        if closed["value"]:
            return
        try:
            while True:
                key, value = results.get_nowait()
                loaded_keys.add(key)
                if key == "sort":
                    sort_label.configure(text=f"CATEGORY: {str(value).upper()}", fg=GREEN)
                    cached_data["sort"] = str(value)
                elif key == "recycle":
                    recycle_label.configure(text=str(value), fg=FG)
                    cached_data["recycle"] = str(value)
                elif key == "dispose":
                    dispose_label.configure(text=str(value), fg="#d4d4d8")
                    cached_data["dispose"] = str(value)
                elif key == "fact":
                    fact_label.configure(text=str(value), fg="#60a5fa")
                    cached_data["fact"] = str(value)

                if len(loaded_keys) >= 4 and not chat_button.winfo_ismapped():
                    chat_button.pack(fill="x", padx=14, pady=(4, 18))
        except queue.Empty:
            pass
        if detail_win.winfo_exists():
            detail_win.after(100, poll_results)

    def on_close():
        closed["value"] = True
        detail_win.destroy()

    detail_win.protocol("WM_DELETE_WINDOW", on_close)
    threading.Thread(target=fetch_data, daemon=True).start()
    detail_win.after(100, poll_results)


def create_card(parent, frame, class_name, confidence):
    card = tk.Frame(
        parent,
        bg=PANEL,
        highlightbackground="#27272a",
        highlightthickness=1,
        padx=12,
        pady=10,
        cursor="hand2",
    )
    card.pack(fill="x", expand=True, padx=14, pady=5)

    image_label = tk.Label(card, bg=PANEL, cursor="hand2")
    image_label.pack(side="left", padx=(0, 12))

    info = tk.Frame(card, bg=PANEL, cursor="hand2")
    info.pack(side="left", fill="both", expand=True)
    name_label = tk.Label(
        info,
        text=class_name.upper(),
        bg=PANEL,
        fg=FG,
        font=("Segoe UI", 11, "bold"),
        anchor="w",
    )
    name_label.pack(anchor="w")
    confidence_label = tk.Label(
        info,
        text=f"Confidence: {confidence * 100:.1f}%",
        bg=PANEL,
        fg=GREEN,
        font=("Segoe UI", 9),
        anchor="w",
    )
    confidence_label.pack(anchor="w", pady=(4, 0))

    def open_details(event=None):
        open_detail_window(class_name, confidence)

    def enter(event=None):
        for widget in (card, image_label, info, name_label, confidence_label):
            widget.configure(bg=PANEL_HOVER)
        card.configure(highlightbackground=GREEN)

    def leave(event=None):
        for widget in (card, image_label, info, name_label, confidence_label):
            widget.configure(bg=PANEL)
        card.configure(highlightbackground="#27272a")

    for widget in (card, image_label, info, name_label, confidence_label):
        widget.bind("<Button-1>", open_details)
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    return image_label


def showUIWin(frame, model_results_pairs, annotated_image=None):
    win = setup_window(
        "EcoVision AI — Detection Results",
        "540x700",
        (420, 500),
    )
    scrollable_frame = setup_scrollable_canvas(win)

    title = tk.Label(
        scrollable_frame,
        text="DETECTED ITEMS",
        bg=BG,
        fg=MUTED,
        font=("Segoe UI", 10, "bold"),
    )
    title.pack(pady=(16, 10))

    if annotated_image is not None:
        preview = tk.Label(scrollable_frame, image=annotated_image, bg=BG)
        preview.image = annotated_image
        preview.pack(fill="x", padx=14, pady=(0, 12))

    found_any = False
    for result, model in model_results_pairs:
        if not result or not hasattr(result, "boxes"):
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].detach().cpu().item())
            class_name = model.names.get(class_id, "unknown")
            confidence = float(box.conf[0].detach().cpu().item())
            crop_image = process_crop(frame, box)
            if crop_image is None:
                continue
            found_any = True
            image_label = create_card(scrollable_frame, frame, class_name, confidence)
            image_label.configure(image=crop_image)
            image_label.image = crop_image

    if not found_any:
        tk.Label(
            scrollable_frame,
            text="No objects detected in this frame.",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 10),
            justify="center",
        ).pack(padx=20, pady=40)

    return win
"""Small window: pick a reference (and optionally your own track), press Analyse.
The work runs in a background thread; the report opens in the browser."""
from __future__ import annotations

import queue
import threading
import traceback
import webbrowser
from pathlib import Path

from . import paths

AUDIO_TYPES = [("Audio", "*.wav *.flac *.mp3 *.aif *.aiff *.ogg"), ("All files", "*.*")]


def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog

    root = tk.Tk()
    root.title("Vokal Analyse")
    root.minsize(560, 330)
    msgs: "queue.Queue[tuple[str, str]]" = queue.Queue()
    ref_var, own_var = tk.StringVar(), tk.StringVar()
    state = {"busy": False}

    frm = ttk.Frame(root, padding=16)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(1, weight=1)

    def picker(row, label, var):
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8)

        def browse():
            f = filedialog.askopenfilename(filetypes=AUDIO_TYPES)
            if f:
                var.set(f)
        ttk.Button(frm, text="Browse...", command=browse).grid(row=row, column=2)

    picker(0, "Reference track", ref_var)
    picker(1, "Your track (optional)", own_var)
    btn = ttk.Button(frm, text="Analyse")
    btn.grid(row=2, column=0, columnspan=3, pady=12, sticky="ew")
    log = tk.Text(frm, height=10, wrap="word", state="disabled")
    log.grid(row=3, column=0, columnspan=3, sticky="nsew")
    frm.rowconfigure(3, weight=1)
    ttk.Label(frm, text=f"Reports are saved in {paths.reports_dir()}", foreground="#777").grid(
        row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def write(text):
        log.configure(state="normal")
        log.insert("end", text + "\n")
        log.see("end")
        log.configure(state="disabled")

    def worker(ref, own):
        try:
            from .separation import StemSeparator
            from .pipeline import run
            sep = StemSeparator(paths.models_dir(), paths.cache_dir(), log=lambda s: msgs.put(("log", s)))
            report = run(ref, own or None, paths.reports_dir(), sep, log=lambda s: msgs.put(("log", s)))
            msgs.put(("done", str(report)))
        except Exception as e:
            msgs.put(("log", "".join(traceback.format_exception_only(type(e), e)).strip()))
            msgs.put(("fail", ""))

    def start():
        if state["busy"]:
            return
        ref, own = ref_var.get().strip(), own_var.get().strip()
        if not ref or not Path(ref).exists():
            write("Pick a reference track first.")
            return
        if own and not Path(own).exists():
            write("Your track was not found.")
            return
        state["busy"] = True
        btn.configure(state="disabled")
        threading.Thread(target=worker, args=(ref, own), daemon=True).start()

    def poll():
        try:
            while True:
                kind, text = msgs.get_nowait()
                if kind == "log":
                    write(text)
                else:
                    state["busy"] = False
                    btn.configure(state="normal")
                    if kind == "done":
                        write("Done. Opening the report in your browser.")
                        webbrowser.open(Path(text).resolve().as_uri())
        except queue.Empty:
            pass
        root.after(150, poll)

    btn.configure(command=start)
    root.after(150, poll)
    root.mainloop()
    return 0

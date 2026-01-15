# main.py
import tkinter as tk
from tkinter import ttk
import threading
from Lessons import LESSONS
from Pose_session import run_pose_session

REMINDERS = [
    "You are doing great!",
    "Adjust your posture for better alignment!",
    "Keep this pose only 5 sec left!",
    "Breathe in... Breathe out...",
]

font1 = "Lora"


# -------------------------
# MESSAGE ROTATOR
# -------------------------
class MessageRotator:
    def __init__(self, widget, messages, interval_ms=3000):
        self.widget = widget
        self.messages = messages or []
        self.interval_ms = interval_ms
        self._idx = 0
        self._after_id = None
        self._running = False

    def start(self):
        if not self.messages or self._running:
            return
        self._running = True
        self._show_current()

    def stop(self):
        self._running = False
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show_current(self):
        if not self._running:
            return
        self.widget.config(text=self.messages[self._idx])
        self._idx = (self._idx + 1) % len(self.messages)
        self._after_id = self.widget.after(self.interval_ms, self._show_current)


# -------------------------
# LESSON PAGE
# -------------------------
class LessonPage(ttk.Frame):
    def __init__(self, parent, lesson, home_callback, get_mode, reminders=None):
        super().__init__(parent, padding=12, style="Lesson.TFrame")
        self.lesson = lesson
        self.home_callback = home_callback
        self.get_mode = get_mode

        self.grid(row=0, column=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)

        ttk.Label(self, text=lesson["title"], style="LessonTitle.TLabel").grid(
            row=0, column=0, pady=(10, 5)
        )
        ttk.Label(
            self,
            text=lesson["description"],
            wraplength=900,
            justify="center"
        ).grid(row=1, column=0, pady=(0, 15))

        self.messages_lbl = ttk.Label(self, text="", style="Messages.TLabel")
        self.messages_lbl.grid(row=2, column=0, pady=(0, 10))

        self.rotator = MessageRotator(self.messages_lbl, reminders or [])

    def on_show(self):
        self.rotator.start()
        threading.Thread(target=self._run_session, daemon=True).start()

    def _run_session(self):
        try:
            run_pose_session(self.lesson["poses"], mode=self.get_mode())

            # --- Show "Lesson Complete!" overlay for 2 seconds ---
            import numpy as np
            import cv2

            display = np.full((480, 640, 3), (211, 177, 211), dtype=np.uint8)  # pink background
            cv2.putText(
                display,
                "Lesson Complete!",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                4,
                cv2.LINE_AA
            )
            cv2.imshow("Yoga Sense", display)
            cv2.waitKey(2000)  # display for 2 seconds
            cv2.destroyAllWindows()

        finally:
            # Always return to home page after finishing lesson
            self.home_callback()


# -------------------------
# HOME PAGE
# -------------------------
class HomePage(ttk.Frame):
    def __init__(self, parent, open_lesson_callback, toggle_mode_callback):
        super().__init__(parent, style="TFrame")
        self.open_lesson_callback = open_lesson_callback
        self.toggle_mode_callback = toggle_mode_callback

        self.grid(row=0, column=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ttk.Label(self, text="Yoga Sense", style="Title.TLabel").grid(row=0, column=1, pady=(30, 20))

        content = ttk.Frame(self, style="TFrame")
        content.grid(row=1, column=1, sticky="n")
        content.grid_columnconfigure(0, weight=1)

        # Lessons cards
        for i, lesson in enumerate(LESSONS):
            card = ttk.Frame(content, padding=20, style="TFrame", relief="raised")
            card.grid(row=i, column=0, pady=15, sticky="ew")
            card.grid_columnconfigure(0, weight=1)

            ttk.Label(card, text=lesson["title"], font=(font1, 16, "bold")).grid(row=0, column=0, pady=(0, 8))
            ttk.Label(card, text=lesson["description"], wraplength=700, justify="center").grid(row=1, column=0, pady=(0, 12))
            ttk.Button(card, text="Open", style="Open.TButton", command=lambda idx=i: self.open_lesson_callback(idx)).grid(row=2, column=0, sticky="ew")

        # Mode toggle
        self.mode_btn = ttk.Button(content, text="Video version", style="Open.TButton", command=self.toggle_mode)
        self.mode_btn.grid(row=len(LESSONS), column=0, pady=(30, 0), sticky="ew")

    def toggle_mode(self):
        self.toggle_mode_callback()
        app = self.toggle_mode_callback.__self__
        self.mode_btn.config(
            text="Coach version" if app.mode == "video" else "Video version"
        )


# -------------------------
# MAIN APP
# -------------------------
class YogaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Yoga Sense")
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda e: self.destroy())
        self.mode = "coach"

        self._setup_style()

        self.container = ttk.Frame(self)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Home page
        self.home_page = HomePage(self.container, self.open_lesson, self.toggle_mode)

        # Lesson pages
        self.lesson_pages = [
            LessonPage(self.container, l, self.show_home, self.get_mode, REMINDERS)
            for l in LESSONS
        ]

        self.show_home()

    def toggle_mode(self):
        self.mode = "video" if self.mode == "coach" else "coach"

    def get_mode(self):
        return self.mode

    def show_home(self):
        self.home_page.tkraise()

    def open_lesson(self, index):
        self.lesson_pages[index].tkraise()
        self.lesson_pages[index].on_show()

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#e4b1d3")
        style.configure("TLabel", background="#e4b1d3")
        style.configure("Title.TLabel", font=(font1, 25, "bold"))
        style.configure("LessonTitle.TLabel", font=(font1, 25, "bold"))
        style.configure("Messages.TLabel", font=("Chewy", 20))
        style.configure("Open.TButton", font=(font1, 12, "bold"))


if __name__ == "__main__":
    YogaApp().mainloop()

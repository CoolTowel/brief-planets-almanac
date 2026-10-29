"""Small Tkinter front end; computation stays in the existing CLI process."""
import argparse
from datetime import date
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread

from .cli import PROJECT_ROOT, parse_utc_offset
from .topo import TopoContext


def generation_command(year, utc_offset, latitude, stars):
    """Validate GUI strings and return an argv list for the existing CLI.

    Latitude is degrees north, the offset is fixed UTC (e.g. +08:00), and
    elevation remains the shared 0 m default. Invalid inputs raise ValueError
    or argparse.ArgumentTypeError. No calculation or file writing occurs.
    """
    topo = TopoContext(int(year), parse_utc_offset(utc_offset.strip()), float(latitude))
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        executable = executable.with_name("python.exe")
    command = [str(executable), "-u", str(PROJECT_ROOT / "main.py"), "all",
               "--year", str(topo.year), f"--utc-offset={utc_offset.strip()}",
               "--latitude", str(topo.latitude_deg)]
    if not stars:
        command.append("--no-stars")
    return command


class AlmanacWindow:
    """Three inputs, a star checkbox and asynchronous generation progress."""

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.process = None
        self.messages = Queue()
        self.output_dir = None
        root.title("日月行星年历")
        root.resizable(False, False)
        frame = ttk.Frame(root, padding=20)
        frame.grid(sticky="nsew")
        self.year = tk.StringVar(value=str(date.today().year))
        self.offset = tk.StringVar(value="+08:00")
        self.latitude = tk.StringVar(value="40")
        self.stars = tk.BooleanVar(value=True)
        self.controls = []
        for row, (label, variable) in enumerate((("年份", self.year),
                                                ("固定 UTC 时区", self.offset),
                                                ("纬度（北纬为正）", self.latitude))):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
            entry = ttk.Entry(frame, textvariable=variable, width=24)
            entry.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=5)
            self.controls.append(entry)
        check = ttk.Checkbutton(frame, text="启用恒星（天狼、织女等）", variable=self.stars)
        check.grid(row=3, column=0, columnspan=2, sticky="w", pady=8)
        self.controls.append(check)
        ttk.Label(frame, text="时区示例：+8、+05:30、-05:00；海拔 0 米。",
                  foreground="#666666").grid(row=4, column=0, columnspan=2, sticky="w")
        self.generate = ttk.Button(frame, text="生成 PNG / PDF", command=self.start)
        self.generate.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(14, 8))
        self.controls.append(self.generate)
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=2, sticky="ew")
        self.status = tk.StringVar(value="准备就绪。全年计算可能需要几分钟。")
        ttk.Label(frame, textvariable=self.status, wraplength=360).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=8)
        self.open_button = ttk.Button(frame, text="打开结果目录", command=self.open_output,
                                      state="disabled")
        self.open_button.grid(row=8, column=0, columnspan=2, sticky="ew")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def start(self):
        """Validate inputs, then run the CLI without blocking Tk's event loop."""
        from tkinter import messagebox

        try:
            command = generation_command(self.year.get(), self.offset.get(),
                                         self.latitude.get(), self.stars.get())
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            self.process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (ValueError, argparse.ArgumentTypeError, OSError) as exc:
            messagebox.showerror("无法开始生成", str(exc), parent=self.root)
            return
        self.output_dir = None
        self.open_button.state(["disabled"])
        for control in self.controls:
            control.state(["disabled"])
        self.status.set("正在计算全年星历……")
        self.progress.start(12)
        Thread(target=self.read_progress, args=(self.process,), daemon=True).start()

    def read_progress(self, process):
        """Queue process output; only the main thread touches Tk widgets."""
        with process.stdout:
            for line in process.stdout:
                self.messages.put(("line", line.strip()))
        self.messages.put(("done", process.wait()))

    def poll(self):
        """Consume progress and completion messages without waiting on the worker."""
        from tkinter import messagebox

        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "line":
                    self.status.set(value)
                    if value.startswith("PNG："):
                        self.output_dir = Path(value.split("：", 1)[1]).parent
                else:
                    self.process = None
                    self.progress.stop()
                    for control in self.controls:
                        control.state(["!disabled"])
                    if value == 0 and self.output_dir is not None:
                        self.status.set("生成完成：PNG 和 PDF 已保存。")
                        self.open_button.state(["!disabled"])
                    else:
                        messagebox.showerror("生成失败", self.status.get(), parent=self.root)
        except Empty:
            pass
        self.root.after(100, self.poll)

    def open_output(self):
        """Open the completed run's folder using the operating system viewer."""
        if self.output_dir is not None:
            if os.name == "nt":
                os.startfile(self.output_dir)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open",
                                  str(self.output_dir)])

    def close(self):
        """Keep an active generation visible until it completes."""
        from tkinter import messagebox

        if self.process is not None:
            messagebox.showinfo("正在生成", "请等待生成完成后关闭窗口。", parent=self.root)
            return
        self.root.destroy()


def launch_gui():
    """Open the lightweight desktop window and return 0 after it closes."""
    import tkinter as tk

    root = tk.Tk()
    AlmanacWindow(root)
    root.mainloop()
    return 0

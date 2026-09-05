#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stgui - a window for starting a scan, for when a terminal is in the way.

It does nothing the command line cannot: pick a folder, scan it, write the HTML
report and open it. The report is the interface; this window exists only so
that starting a scan does not require typing a command.

    python stgui.py            # Windows: pythonw stgui.py, to skip the console

No third-party dependencies - tkinter, threading and webbrowser are all
standard library. Plain tkinter has no drag and drop (that needs tkdnd, which
would be a dependency), so a folder arrives through Browse or by typing a path.

Scanning runs on a worker thread and reports through a queue; Tk is touched
only from the main thread.
"""
from __future__ import annotations

import queue
import sys
import tempfile
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, ttk

import report_html
import stinspect
from i18n import L

# Every message key this window uses. tools/self_check.py reads this list to
# check both language catalogues carry them.
GUI_KEYS = (
    "gui_title", "gui_target", "gui_output", "gui_output_default",
    "gui_output_default_path", "gui_browse", "gui_recursive", "gui_meta",
    "gui_keys", "gui_lang", "gui_scan", "gui_open", "gui_cancel", "gui_pick_target",
    "gui_pick_output", "gui_need_target", "gui_collecting", "gui_progress",
    "gui_none", "gui_done", "gui_cancelled", "gui_failed",
)


def default_report_path(target: Path) -> Path:
    """Where a report goes when the output box is left empty.

    The name comes from stinspect so it matches what `--html auto` writes; only
    the folder differs. A temporary one here: this report is opened at once, and
    a stable name per scanned folder means a rescan replaces the file the
    browser already has open instead of leaving a trail behind.
    """
    return (Path(tempfile.gettempdir())
            / stinspect.default_report_name(target))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.lang = tk.StringVar(value="en")
        self.target = tk.StringVar()
        self.output = tk.StringVar()
        self.recursive = tk.BooleanVar(value=True)
        self.full_meta = tk.BooleanVar(value=False)
        self.show_keys = tk.BooleanVar(value=False)

        self.events: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.labels: dict[str, tk.Widget] = {}

        self._build()
        self._retranslate()
        self._target_changed()
        self.after(60, self._drain)

    # -- layout ------------------------------------------------------------
    def _build(self):
        self.columnconfigure(0, weight=1)
        frame = ttk.Frame(self, padding=12)
        frame.grid(sticky="nsew")
        frame.columnconfigure(1, weight=1)

        self.labels["target"] = ttk.Label(frame)
        self.labels["target"].grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.target, width=52).grid(
            row=0, column=1, sticky="ew", padx=6)
        self.labels["browse_target"] = ttk.Button(frame, command=self._pick_target)
        self.labels["browse_target"].grid(row=0, column=2)

        self.labels["output"] = ttk.Label(frame)
        self.labels["output"].grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.output).grid(
            row=1, column=1, sticky="ew", padx=6)
        self.labels["browse_output"] = ttk.Button(frame, command=self._pick_output)
        self.labels["browse_output"].grid(row=1, column=2)

        self.labels["output_default"] = ttk.Label(frame, foreground="#666")
        self.labels["output_default"].grid(row=2, column=1, sticky="w", padx=6)

        opts = ttk.Frame(frame)
        opts.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 4))
        self.labels["recursive"] = ttk.Checkbutton(opts, variable=self.recursive)
        self.labels["recursive"].grid(row=0, column=0, padx=(0, 14))
        self.labels["meta"] = ttk.Checkbutton(opts, variable=self.full_meta)
        self.labels["meta"].grid(row=0, column=1, padx=(0, 14))
        self.labels["keys"] = ttk.Checkbutton(opts, variable=self.show_keys)
        self.labels["keys"].grid(row=0, column=2, padx=(0, 14))
        self.labels["lang"] = ttk.Label(opts)
        self.labels["lang"].grid(row=0, column=3, padx=(0, 6))
        box = ttk.Combobox(opts, textvariable=self.lang, values=("en", "ja"),
                           state="readonly", width=5)
        box.grid(row=0, column=4)
        box.bind("<<ComboboxSelected>>", lambda _e: self._retranslate())
        # Say where an empty output box will actually write, as soon as there
        # is a folder to name the file after, and offer to open whatever report
        # is already sitting at that path.
        self.target.trace_add("write", lambda *_: self._target_changed())
        self.output.trace_add("write", lambda *_: self._target_changed())

        self.bar = ttk.Progressbar(frame, mode="determinate")
        self.bar.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 4))

        self.status = ttk.Label(frame, anchor="w")
        self.status.grid(row=5, column=0, columnspan=3, sticky="ew")

        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=3, sticky="e", pady=(10, 0))
        self.labels["open"] = ttk.Button(buttons, command=self._open)
        self.labels["open"].grid(row=0, column=0, padx=(0, 8))
        self.run = ttk.Button(buttons, command=self._toggle)
        self.run.grid(row=0, column=1)

    def _retranslate(self):
        lang = self.lang.get()
        self.title(L(lang, "gui_title"))
        self.labels["target"].config(text=L(lang, "gui_target"))
        self.labels["output"].config(text=L(lang, "gui_output"))
        self._show_default()
        for key in ("browse_target", "browse_output"):
            self.labels[key].config(text=L(lang, "gui_browse"))
        self.labels["recursive"].config(text=L(lang, "gui_recursive"))
        self.labels["meta"].config(text=L(lang, "gui_meta"))
        self.labels["keys"].config(text=L(lang, "gui_keys"))
        self.labels["lang"].config(text=L(lang, "gui_lang"))
        self.labels["open"].config(text=L(lang, "gui_open"))
        self.run.config(text=L(lang, "gui_cancel" if self.worker else "gui_scan"))

    def _target_changed(self):
        self._show_default()
        self._refresh_open()

    def _show_default(self):
        lang = self.lang.get()
        target = self.target.get().strip()
        text = (L(lang, "gui_output_default_path",
                  path=default_report_path(Path(target)))
                if target else L(lang, "gui_output_default"))
        self.labels["output_default"].config(text=text)

    def report_path(self):
        """Where a report would be, whether or not one has been written yet."""
        out = self.output.get().strip()
        if out:
            return Path(out)
        target = self.target.get().strip()
        return default_report_path(Path(target)) if target else None

    def _refresh_open(self):
        """Open is live whenever a report is actually sitting at that path -
        including one written by an earlier run, so a closed tab is one click
        back rather than another scan."""
        path = self.report_path()
        ready = path is not None and path.is_file()
        self.labels["open"].config(state="normal" if ready else "disabled")

    # -- actions -----------------------------------------------------------
    def _pick_target(self):
        path = filedialog.askdirectory(title=L(self.lang.get(), "gui_pick_target"))
        if path:
            self.target.set(path)

    def _pick_output(self):
        current = self.output.get().strip()
        target = self.target.get().strip()
        # Offer the name that would be used anyway, so the dialog never opens
        # with an empty file name box.
        if current:
            initial_dir, initial_file = str(Path(current).parent), Path(current).name
        else:
            initial_dir = ""
            initial_file = (stinspect.default_report_name(target) if target
                            else "stinspect-report.html")
        path = filedialog.asksaveasfilename(
            title=L(self.lang.get(), "gui_pick_output"),
            defaultextension=".html",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[("HTML", "*.html")])
        if path:
            self.output.set(path)

    def _open(self):
        path = self.report_path()
        if path and path.is_file():
            webbrowser.open(path.resolve().as_uri())

    def _toggle(self):
        if self.worker:
            self.stop.set()
            return
        lang = self.lang.get()
        target = self.target.get().strip()
        if not target or not Path(target).exists():
            self.status.config(text=L(lang, "gui_need_target"))
            return
        out = self.output.get().strip()
        out_path = Path(out) if out else default_report_path(Path(target))

        self.stop.clear()
        self.bar.config(value=0, maximum=1)
        self.status.config(text=L(lang, "gui_collecting"))
        self.worker = threading.Thread(
            target=self._work,
            args=(Path(target), out_path, lang, self.recursive.get(),
                  self.full_meta.get(), self.show_keys.get()),
            daemon=True)
        self.worker.start()
        self._retranslate()

    def _work(self, target, out_path, lang, recursive, full_meta, show_keys):
        """Runs off the main thread: only the queue may cross back."""
        try:
            files = stinspect.collect_files([str(target)], recursive, lang)
            self.events.put(("total", len(files)))
            if not files:
                self.events.put(("none", None))
                return
            results = []
            for i, path in enumerate(files, 1):
                if self.stop.is_set():
                    self.events.put(("cancelled", None))
                    return
                results.append(stinspect.analyze(path))
                self.events.put(("progress", (i, path.name)))
            page = stinspect.build_page(results, lang, full_meta=full_meta,
                                        show_keys=show_keys)
            report_html.write_html(page, out_path)
            self.events.put(("done", (len(results), out_path)))
        except Exception as exc:                      # noqa: BLE001 - shown to the user
            self.events.put(("failed", exc))

    def _drain(self):
        lang = self.lang.get()
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "total":
                    self.bar.config(maximum=max(payload, 1), value=0)
                elif kind == "progress":
                    done, name = payload
                    self.bar.config(value=done)
                    self.status.config(text=L(lang, "gui_progress",
                                              done=done,
                                              total=int(self.bar["maximum"]),
                                              name=name))
                elif kind == "none":
                    self._finish(L(lang, "gui_none"))
                elif kind == "cancelled":
                    self._finish(L(lang, "gui_cancelled"))
                elif kind == "failed":
                    self._finish(L(lang, "gui_failed", err=payload))
                elif kind == "done":
                    n, out_path = payload
                    self._finish(L(lang, "gui_done", n=n, path=out_path))
                    webbrowser.open(Path(out_path).resolve().as_uri())
        except queue.Empty:
            pass
        self.after(60, self._drain)

    def _finish(self, message):
        self.worker = None
        self.status.config(text=message)
        self._retranslate()
        self._refresh_open()


def main():
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

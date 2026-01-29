"""Trig Sketching Trainer (Sine/Cosine/Tangent/Secant/Cosecant/Cotangent)

Features
- Two quiz modes:
  1) Equation -> you sketch; use "Reveal Answer" to view the graph.
  2) Graph -> you write the equation; use "Reveal Answer" to compare.
- Toggles:
  - Include Sine / Cosine / Tangent / Secant / Cosecant / Cotangent
  - Quiz mode: Equation->Graph, Graph->Equation, or Random
  - Random prompt type (Equation given vs Graph given) is implicit in mode=Random.
- Randomized parameters: amplitude, period, phase shift, vertical shift.

Run:
  python src/main_frame.py

Requires: Python 3.10+, matplotlib
  pip install matplotlib

Notes
- This is a study tool; use "Reveal Answer" to compare your work.
"""

from __future__ import annotations

import math
import random
import tkinter as tk
from tkinter import ttk, messagebox

from generate_question import generate_question
from trig_math import PI, TrigFunction, format_pi_tick_math

# --- Dependencies (numpy + matplotlib) ---
# If VS Code underlines these imports, it means your selected Python interpreter
# does not have the packages installed.
try:
    import numpy as np
except Exception as e:
    raise SystemExit(
        "Missing dependency: numpy. Install with:  python -m pip install numpy"
        f"Details: {e}"
    )

try:
    import matplotlib
    # Force a Tk-compatible backend (prevents 'graph doesn't appear' issues).
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib import ticker
except Exception as e:
    raise SystemExit(
        "Missing dependency: matplotlib (Tk backend). Install with:  python -m pip install matplotlib"
        f"Details: {e}"
    )


# ----------------------------
# UI
# ----------------------------

class TrigTrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trig Sketching Trainer")
        self.geometry("1100x680")

        # State
        self.include_sin = tk.BooleanVar(value=True)
        self.include_cos = tk.BooleanVar(value=True)
        self.include_tan = tk.BooleanVar(value=False)
        self.include_cot = tk.BooleanVar(value=False)
        self.include_sec = tk.BooleanVar(value=False)
        self.include_csc = tk.BooleanVar(value=False)
        self.theme = tk.StringVar(value="Dark")
        self.periods_to_display = tk.StringVar(value="1")
        self.anchor_mode = tk.StringVar(value="C")
        self.show_equiv_forms = tk.BooleanVar(value=True)
        self.show_guides_setting = tk.BooleanVar(value=True)

        # Randomization toggles (A,B,C,D)
        self.rand_A = tk.BooleanVar(value=True)  # amplitude
        self.rand_B = tk.BooleanVar(value=True)  # frequency/period
        self.rand_C = tk.BooleanVar(value=True)  # phase shift
        self.rand_D = tk.BooleanVar(value=True)  # vertical shift

        self.mode = tk.StringVar(value="Random")  # Equation->Graph, Graph->Equation, Random

        self.current: TrigFunction | None = None
        self.prompt_type: str | None = None  # 'equation' or 'graph'

        # Display flags
        self.graph_visible = False
        self.show_guides = False  # midline + amplitude bounds when answer is revealed
        self.answer_revealed = False  # toggles Reveal/Hide behavior

        # Plot setup
        self.fig = Figure(figsize=(6.2, 4.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True, alpha=0.25)

        self.canvas = None  # created after plot_frame exists

        self.style = ttk.Style(self)
        self._build_layout()
        self._apply_theme()
        self._new_question()

    def _build_layout(self):
        # Left control panel
        left = ttk.Frame(self, padding=12)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Settings", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        # Function toggles
        fbox = ttk.LabelFrame(left, text="Include", padding=10)
        fbox.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(fbox, text="Sine (sin)", variable=self.include_sin, command=self._new_question).pack(anchor="w")
        ttk.Checkbutton(fbox, text="Cosine (cos)", variable=self.include_cos, command=self._new_question).pack(anchor="w")
        ttk.Checkbutton(fbox, text="Tangent (tan)", variable=self.include_tan, command=self._new_question).pack(anchor="w")
        ttk.Checkbutton(fbox, text="Cotangent (cot)", variable=self.include_cot, command=self._new_question).pack(anchor="w")
        ttk.Checkbutton(fbox, text="Secant (sec)", variable=self.include_sec, command=self._new_question).pack(anchor="w")
        ttk.Checkbutton(fbox, text="Cosecant (csc)", variable=self.include_csc, command=self._new_question).pack(anchor="w")

        # Parameter randomization toggles
        pbox = ttk.LabelFrame(left, text="Randomize Parameters", padding=10)
        pbox.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(pbox, text="A (Amplitude)", variable=self.rand_A, command=self._new_question).pack(anchor="w")
        ttk.Checkbutton(pbox, text="B (Period / Stretch)", variable=self.rand_B, command=self._new_question).pack(anchor="w")
        ttk.Checkbutton(pbox, text="C (Horizontal Shift)", variable=self.rand_C, command=self._new_question).pack(anchor="w")
        ttk.Checkbutton(pbox, text="D (Vertical Shift)", variable=self.rand_D, command=self._new_question).pack(anchor="w")

        # Mode selector
        mbox = ttk.LabelFrame(left, text="Mode", padding=10)
        mbox.pack(fill=tk.X, pady=(10, 0))
        modes = ["Equation -> Graph", "Graph -> Equation", "Random"]
        for m in modes:
            ttk.Radiobutton(mbox, text=m, value=m, variable=self.mode, command=self._new_question).pack(anchor="w")

        # View selector (periods/anchor)
        vbox = ttk.LabelFrame(left, text="View", padding=10)
        vbox.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(vbox, text="Periods to display").pack(anchor="w")
        self.periods_combo = ttk.Combobox(
            vbox,
            values=["1", "2", "3"],
            textvariable=self.periods_to_display,
            state="readonly",
            width=5,
        )
        self.periods_combo.pack(anchor="w", pady=(4, 8))
        self.periods_combo.bind("<<ComboboxSelected>>", self._on_view_change)

        ttk.Label(vbox, text="Anchor").pack(anchor="w")
        ttk.Radiobutton(
            vbox,
            text="Anchor to horizontal shift (C)",
            value="C",
            variable=self.anchor_mode,
            command=self._on_view_change,
        ).pack(anchor="w")
        ttk.Radiobutton(
            vbox,
            text="Anchor to 0",
            value="0",
            variable=self.anchor_mode,
            command=self._on_view_change,
        ).pack(anchor="w")

        ttk.Checkbutton(
            vbox,
            text="Show equivalent forms (+sin/-sin/+cos/-cos)",
            variable=self.show_equiv_forms,
            command=self._on_view_change,
        ).pack(anchor="w", pady=(6, 0))
        ttk.Checkbutton(
            vbox,
            text="Show guides (midline and amplitude bounds)",
            variable=self.show_guides_setting,
            command=self._on_view_change,
        ).pack(anchor="w")

        # Theme selector
        tbox = ttk.LabelFrame(left, text="Theme", padding=10)
        tbox.pack(fill=tk.X, pady=(10, 0))
        ttk.Radiobutton(tbox, text="Dark", value="Dark", variable=self.theme, command=self._apply_theme).pack(anchor="w")
        ttk.Radiobutton(tbox, text="Light", value="Light", variable=self.theme, command=self._apply_theme).pack(anchor="w")

        # Controls
        cbox = ttk.Frame(left)
        cbox.pack(fill=tk.X, pady=(12, 0))

        ttk.Button(cbox, text="New Question", command=self._new_question).pack(fill=tk.X)
        self.reveal_btn = ttk.Button(cbox, text="Reveal Answer", command=self._toggle_answer)
        self.reveal_btn.pack(fill=tk.X, pady=(6, 0))

        # Right panel: prompt + plot
        right = ttk.Frame(self, padding=12)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.prompt_lbl = ttk.Label(right, text="", font=("Segoe UI", 14, "bold"), wraplength=760, justify="left")
        self.prompt_lbl.pack(anchor="w")

        plot_frame = ttk.Frame(right)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # IMPORTANT: create the matplotlib canvas *inside* plot_frame.
        # Creating it with master=self and packing into another frame can result in a blank area.
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()


    def _apply_theme(self):
        if self.theme.get() == "Dark":
            colors = {
                "bg": "#1f2327",
                "text": "#e6e6e6",
                "muted_text": "#b8b8b8",
                "button": "#2c3137",
                "button_active": "#363c43",
                "entry_bg": "#1a1d21",
                "entry_fg": "#f0f0f0",
                "plot_bg": "#1a1d21",
                "axis_text": "#e6e6e6",
                "axis_line": "#cfd4d8",
                "grid_major": "#3a3f45",
                "grid_minor": "#2c3136",
                "curve": "#4ea3ff",
                "guide": "#ff9f43",
                "text_box": "#2c3137",
                "spine": "#2a2f34",
            }
        else:
            colors = {
                "bg": "#f2f2f2",
                "text": "#1d1d1d",
                "muted_text": "#4a4a4a",
                "button": "#e6e6e6",
                "button_active": "#d9d9d9",
                "entry_bg": "#ffffff",
                "entry_fg": "#1d1d1d",
                "plot_bg": "#ffffff",
                "axis_text": "#1d1d1d",
                "axis_line": "#000000",
                "grid_major": "#bdbdbd",
                "grid_minor": "#dedede",
                "curve": "#1f77b4",
                "guide": "#d97706",
                "text_box": "#f0f0f0",
                "spine": "#b0b0b0",
            }

        self.colors = colors

        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.configure(bg=colors["bg"])
        self.style.configure("TFrame", background=colors["bg"])
        self.style.configure("TLabelframe", background=colors["bg"], foreground=colors["text"])
        self.style.configure("TLabelframe.Label", background=colors["bg"], foreground=colors["text"])
        self.style.configure("TLabel", background=colors["bg"], foreground=colors["text"])
        self.style.configure("TButton", background=colors["button"], foreground=colors["text"])
        self.style.map(
            "TButton",
            background=[("active", colors["button_active"]), ("disabled", colors["bg"])],
            foreground=[("disabled", colors["muted_text"])],
        )
        self.style.configure(
            "TEntry",
            fieldbackground=colors["entry_bg"],
            foreground=colors["entry_fg"],
            background=colors["bg"],
        )
        self.style.map(
            "TEntry",
            fieldbackground=[("disabled", colors["bg"])],
            foreground=[("disabled", colors["muted_text"])],
        )
        self.style.configure(
            "TCombobox",
            fieldbackground=colors["entry_bg"],
            foreground=colors["entry_fg"],
            background=colors["bg"],
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", colors["entry_bg"])],
            foreground=[("readonly", colors["entry_fg"])],
        )
        self.style.configure("TCheckbutton", background=colors["bg"], foreground=colors["text"])
        self.style.configure("TRadiobutton", background=colors["bg"], foreground=colors["text"])

        if self.current is not None and self.prompt_type is not None:
            self._refresh_plot()
        else:
            self.fig.patch.set_facecolor(colors["plot_bg"])
            self.ax.set_facecolor(colors["plot_bg"])
            self.canvas.draw()

    def _refresh_plot(self):
        if not self.current or not self.prompt_type:
            return
        if self.prompt_type == "equation":
            if self.answer_revealed:
                self._show_graph(show_equation=False)
            else:
                self._hide_graph(show_equation=True)
        else:
            self._show_graph(show_equation=self.answer_revealed)

    def _style_axes(self):
        colors = self.colors
        self.fig.patch.set_facecolor(colors["plot_bg"])
        self.ax.set_facecolor(colors["plot_bg"])
        self.ax.tick_params(colors=colors["axis_text"], which="both")
        for spine in self.ax.spines.values():
            spine.set_color(colors["spine"])

    def _get_periods_to_display(self) -> int:
        try:
            periods = int(self.periods_to_display.get())
        except (TypeError, ValueError):
            periods = 1
        return max(1, min(3, periods))

    def _equivalent_forms_text(self, tf: TrigFunction) -> str:
        if tf.kind not in ("sin", "cos"):
            return ""
        if abs(tf.B) < 1e-12:
            return ""

        sin_shift = tf.C if tf.kind == "sin" else tf.C - (PI / (2 * tf.B))
        forms = [
            ("+sin", TrigFunction("sin", tf.A, tf.B, sin_shift, tf.D)),
            ("-sin", TrigFunction("sin", tf.A, tf.B, sin_shift - (PI / tf.B), tf.D)),
            ("+cos", TrigFunction("cos", tf.A, tf.B, sin_shift + (PI / (2 * tf.B)), tf.D)),
            ("-cos", TrigFunction("cos", -tf.A, tf.B, sin_shift - (PI / (2 * tf.B)), tf.D)),
        ]

        lines = []
        for label, form in forms:
            eq = form.pretty()
            lines.append(f"$\\mathrm{{{label}}}\\; {eq}$")
        return "\n".join(lines)

    def _on_view_change(self, *_):
        self._refresh_plot()

    def _new_question(self):
        if not (
            self.include_sin.get()
            or self.include_cos.get()
            or self.include_tan.get()
            or self.include_cot.get()
            or self.include_sec.get()
            or self.include_csc.get()
        ):
            messagebox.showwarning(
                "No functions selected",
                "Select at least one of sine/cosine/tangent/cotangent/secant/cosecant.",
            )
            self.include_sin.set(True)

        self.current = generate_question(
            self.include_sin.get(),
            self.include_cos.get(),
            self.include_tan.get(),
            self.include_cot.get(),
            self.include_sec.get(),
            self.include_csc.get(),
            self.rand_A.get(),
            self.rand_B.get(),
            self.rand_C.get(),
            self.rand_D.get(),
        )

        # Reset flags for new question
        self.show_guides = False
        self.answer_revealed = False
        self.reveal_btn.configure(text="Reveal Answer")

        # Decide prompt type
        mode = self.mode.get()
        if mode == "Equation -> Graph":
            self.prompt_type = "equation"
        elif mode == "Graph -> Equation":
            self.prompt_type = "graph"
        else:
            self.prompt_type = random.choice(["equation", "graph"])

        self._hide_graph(show_equation=(self.prompt_type == "equation"))

        if self.prompt_type == "equation":
            self.prompt_lbl.configure(
                text=(
                    "Sketch this by hand. The equation is shown on the grid"
                    "Use 'Reveal Answer' to show/hide the graph."
                )
            )
        else:
            self.prompt_lbl.configure(
                text=(
                    "Write the equation for this graph.\n"
                    "(Tip: identify A, period, phase shift, and vertical shift.)"
                )
            )
            self._show_graph()


    def _plot_function(
        self,
        tf: TrigFunction,
        title: str | None = None,
        show_guides: bool = False,
        show_equation: bool = False,
        show_equiv: bool = False,
    ):
        self.ax.clear()
        self._style_axes()

        # ---- Period window anchored for sin/cos view settings ----
        P = tf.period
        if tf.kind in ("sin", "cos", "tan", "cot", "sec", "csc"):
            periods = self._get_periods_to_display()
            if self.anchor_mode.get() == "0":
                start = 0.0
            else:
                start = tf.C
            end = start + periods * P
        else:
            start = tf.C
            end = start + P

        xs = np.linspace(start, end, 1600)
        ys = tf.y(xs)
        if tf.kind in ("tan", "cot", "sec", "csc"):
            # Avoid drawing across asymptotes.
            arg = tf.B * (xs - tf.C)
            if tf.kind in ("tan", "sec"):
                denom = np.cos(arg)
            else:
                denom = np.sin(arg)
            ys = ys.copy()
            ys[np.abs(denom) < 0.08] = np.nan
            y_limit = 12.0 * max(1.0, abs(tf.A))
            ys[np.abs(ys - tf.D) > y_limit] = np.nan

        # ---- Plot ----
        # Distinct graph styling so it doesn't blend with the axes
        self.ax.plot(xs, ys, linewidth=2.6, color=self.colors["curve"])

        # Precompute y-limits for consistent axis-line decisions.
        if tf.kind in ("tan", "cot", "sec", "csc"):
            y_span = max(4.0, 3.0 * abs(tf.A))
            y_min = tf.D - y_span
            y_max = tf.D + y_span
        else:
            ypad = max(0.75, 0.20 * (2 * abs(tf.A)))
            y_min = tf.D - abs(tf.A) - ypad
            y_max = tf.D + abs(tf.A) + ypad

        # Axes lines (helps see intercepts clearly) — thin black so they don't dominate
        axis_style = dict(color=self.colors["axis_line"], linewidth=1.0, alpha=0.85)
        if start <= 0 <= end:
            self.ax.axvline(0, **axis_style)
        if y_min <= 0 <= y_max:
            self.ax.axhline(0, **axis_style)

        # Helpful reference lines (only after Reveal Answer): midline + amplitude bounds
        if show_guides and abs(tf.D) > 1e-12:
            # Brighter, attention-grabbing guide lines
            guide_style = dict(linestyle=":", linewidth=2.0, color=self.colors["guide"], alpha=0.95)
            self.ax.axhline(tf.D, **guide_style)  # midline (now dotted too)
            if tf.kind in ("sin", "cos"):
                self.ax.axhline(tf.D + abs(tf.A), **guide_style)
                self.ax.axhline(tf.D - abs(tf.A), **guide_style)

        # Limits: full periods on x
        self.ax.set_xlim((start, end))

        # Y-limits: tight but readable
        self.ax.set_ylim((y_min, y_max))

        # Labels/title
        self.ax.set_xlabel("x", color=self.colors["axis_text"])
        self.ax.set_ylabel("y", color=self.colors["axis_text"])
        if title:
            self.ax.set_title(title, color=self.colors["axis_text"])

        if show_equation:
            self.ax.text(
                0.02,
                0.98,
                tf.pretty_math(),
                transform=self.ax.transAxes,
                va="top",
                ha="left",
                fontsize=16,
                color=self.colors["axis_text"],
                bbox=dict(boxstyle="round", alpha=0.15, facecolor=self.colors["text_box"], edgecolor="none"),
            )

        if show_equiv and tf.kind in ("sin", "cos"):
            forms_text = self._equivalent_forms_text(tf)
            if forms_text:
                self.ax.text(
                    0.98,
                    0.98,
                    forms_text,
                    transform=self.ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=11,
                    color=self.colors["axis_text"],
                    bbox=dict(boxstyle="round", alpha=0.15, facecolor=self.colors["text_box"], edgecolor="none"),
                )

        # ---- "Zoomed in" grid like class graphs ----
        # X ticks: 8 boxes per period (like 0, π/4, π/2, ... when P=2π)
        major_step_x = P / 8
        minor_step_x = major_step_x / 2

        major_ticks = np.arange(start, end + 1e-9, major_step_x)
        minor_ticks = np.arange(start, end + 1e-9, minor_step_x)

        self.ax.set_xticks(major_ticks)
        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_xticklabels(
            [format_pi_tick_math(v) for v in major_ticks],
            rotation=0,
            color=self.colors["axis_text"],
        )

        # Y ticks: major every 1, minor every 0.5
        y_min, y_max = self.ax.get_ylim()
        y_major = 1.0
        y_minor = 0.5

        y0 = math.floor(y_min / y_major) * y_major
        y1 = math.ceil(y_max / y_major) * y_major
        y_major_ticks = np.arange(y0, y1 + 1e-9, y_major)
        y_minor_ticks = np.arange(y0, y1 + 1e-9, y_minor)

        self.ax.set_yticks(y_major_ticks)
        self.ax.set_yticks(y_minor_ticks, minor=True)

        # Grid: show visible boxes
        self.ax.grid(True, which="major", linewidth=1.0, alpha=0.35, color=self.colors["grid_major"])
        self.ax.grid(True, which="minor", linewidth=0.6, alpha=0.2, color=self.colors["grid_minor"])

        self.canvas.draw()

    def _show_graph(self, show_equation: bool = False):
        if not self.current:
            return
        self.graph_visible = True
        # In Equation->Graph prompts, revealing the graph is the answer.
        # In Graph->Equation prompts, keep guides/equation hidden until revealed.
        if self.current.kind in ("sin", "cos"):
            want_guides = self.show_guides_setting.get() and self.show_guides
        else:
            want_guides = self.show_guides
        want_eq = show_equation
        want_equiv = (
            self.answer_revealed
            and self.show_equiv_forms.get()
            and self.current.kind in ("sin", "cos")
        )
        self._plot_function(
            self.current,
            title="Target graph",
            show_guides=want_guides,
            show_equation=want_eq,
            show_equiv=want_equiv,
        )

    def _hide_graph(self, show_equation: bool = False):
        self.graph_visible = False
        self.ax.clear()
        self._style_axes()

        # Default "blank" view still looks like a graphing grid
        self.ax.set_xlabel("x", color=self.colors["axis_text"])
        self.ax.set_ylabel("y", color=self.colors["axis_text"])
        self.ax.set_title("(Graph hidden)", color=self.colors["axis_text"])

        if show_equation and self.current is not None:
            # Typeset equation inside the plot area.
            self.ax.text(
                0.02,
                0.98,
                self.current.pretty_math(),
                transform=self.ax.transAxes,
                va="top",
                ha="left",
                fontsize=16,
                color=self.colors["axis_text"],
                bbox=dict(boxstyle="round", alpha=0.15, facecolor=self.colors["text_box"], edgecolor="none"),
            )

        # Show y-intercept / origin area clearly
        self.ax.set_xlim((-math.pi, math.pi))
        self.ax.set_ylim((-2, 2))

        axis_style = dict(color=self.colors["axis_line"], linewidth=1.0, alpha=0.85)
        self.ax.axhline(0, **axis_style)
        self.ax.axvline(0, **axis_style)

        # Grid boxes
        self.ax.set_xticks(np.arange(-math.pi, math.pi + 1e-9, math.pi / 4))
        self.ax.set_xticks(np.arange(-math.pi, math.pi + 1e-9, math.pi / 8), minor=True)
        self.ax.set_xticklabels(
            [format_pi_tick_math(v) for v in self.ax.get_xticks()],
            rotation=0,
            color=self.colors["axis_text"],
        )

        self.ax.set_yticks(np.arange(-2, 2.0001, 1.0))
        self.ax.set_yticks(np.arange(-2, 2.0001, 0.5), minor=True)

        self.ax.grid(True, which="major", linewidth=1.0, alpha=0.35, color=self.colors["grid_major"])
        self.ax.grid(True, which="minor", linewidth=0.6, alpha=0.2, color=self.colors["grid_minor"])

        self.canvas.draw()

    def _toggle_answer(self):
        if not self.current:
            return

        if not self.answer_revealed:
            # --- Reveal ---
            self.answer_revealed = True
            self.reveal_btn.configure(text="Hide Answer")

            if self.prompt_type == "equation":
                # Answer is the graph.
                self.show_guides = True
                self._show_graph(show_equation=False)
            else:
                # prompt_type == 'graph': answer is the equation.
                self.show_guides = True
                self._show_graph(show_equation=True)
        else:
            # --- Hide ---
            self.answer_revealed = False
            self.reveal_btn.configure(text="Reveal Answer")

            if self.prompt_type == "equation":
                # Go back to equation-only grid.
                self.show_guides = False
                self._hide_graph(show_equation=True)
            else:
                # Keep the graph (since it's the prompt), but remove equation/guides.
                self.show_guides = False
                self._show_graph(show_equation=False)


def main():
    app = TrigTrainerApp()
    app.mainloop()


if __name__ == "__main__":
    main()

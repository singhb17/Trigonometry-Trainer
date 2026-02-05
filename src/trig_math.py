from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PI = math.pi


def trim_float(x: float) -> str:
    # Compact floats like 2.0 -> 2, 0.5 -> 0.5
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    s = f"{x:.6g}"
    return s


def sec(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.cos(x)


def csc(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sin(x)


def cot(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.tan(x)


def format_pi_multiple(x: float) -> str | None:
    """If x is close to a common multiple of pi/6, pi/4, pi/3, pi/2, pi, return a string."""
    # Try rational multiples of pi with denominators up to 12
    for den in (24, 16, 12, 8, 6, 4, 3, 2, 1):
        for num in range(-24, 25):
            val = (num / den) * PI
            if abs(x - val) < 1e-9:
                if num == 0:
                    return "0"
                # simplify num/den
                g = math.gcd(abs(num), den)
                n = num // g
                d = den // g

                sign = "-" if n < 0 else ""
                n_abs = abs(n)
                if d == 1:
                    if n_abs == 1:
                        return f"{sign}\\pi"
                    return f"{sign}{n_abs}\\pi"
                # d > 1
                if n_abs == 1:
                    return f"{sign}\\pi/{d}"
                return f"{sign}{n_abs}\\pi/{d}"
    return None


def format_pi_tick(x: float) -> str:
    s = format_pi_multiple(x)
    if s is not None:
        return s
    return trim_float(x)


def format_pi_tick_math(x: float) -> str:
    # Mathtext wrapper for nicer tick formatting in matplotlib.
    return f"${format_pi_tick(x)}$"


@dataclass(frozen=True)
class TrigFunction:
    kind: str  # 'sin', 'cos', 'tan', 'cot', 'sec', or 'csc'
    A: float
    B: float
    C: float
    D: float

    def y(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "sin":
            return self.A * np.sin(self.B * (x - self.C)) + self.D
        if self.kind == "cos":
            return self.A * np.cos(self.B * (x - self.C)) + self.D
        if self.kind == "tan":
            return self.A * np.tan(self.B * (x - self.C)) + self.D
        if self.kind == "cot":
            return self.A * cot(self.B * (x - self.C)) + self.D
        if self.kind == "sec":
            return self.A * sec(self.B * (x - self.C)) + self.D
        return self.A * csc(self.B * (x - self.C)) + self.D

    @property
    def period(self) -> float:
        if self.kind in ("tan", "cot"):
            return math.pi / abs(self.B)
        return 2 * math.pi / abs(self.B)

    def pretty(self) -> str:
        # y = A sin(B(x - C)) + D  (or cos)
        # Keep it readable and consistent.
        kind = self.kind
        A = self.A
        B = self.B
        C = self.C
        D = self.D

        def fmt(n: float) -> str:
            # Prefer exact-looking fractions for common pi multiples
            s = format_pi_multiple(n)
            return s if s is not None else trim_float(n)

        A_str = "" if abs(A - 1) < 1e-12 else ("-" if abs(A + 1) < 1e-12 else trim_float(A))

        # B as float (often a rational like 0.5, 1, 2, 3)
        B_str = "" if abs(B - 1) < 1e-12 else ("-" if abs(B + 1) < 1e-12 else trim_float(B))

        # (x - C) formatting
        if abs(C) < 1e-12:
            inner = "x"
        else:
            # x - C
            # if C negative => x + |C|
            sign = "-" if C > 0 else "+"
            inner = f"x {sign} {fmt(abs(C))}"

        base = f"{A_str}{kind}({B_str}({inner}))" if B_str else f"{A_str}{kind}({inner})"

        if abs(D) < 1e-12:
            return f"y = {base}"
        sign = "+" if D > 0 else "-"
        return f"y = {base} {sign} {fmt(abs(D))}"

    def pretty_math(self) -> str:
        """Math-looking string for matplotlib (mathtext).

        Tkinter labels are plain text, but matplotlib will render $...$ nicely.
        This keeps things simple: it uses \\pi mathtext and '/' fractions.
        """
        s = self.pretty().replace("y = ", "")
        # Wrap in mathtext to get typeset italics / spacing.
        # (Even without LaTeX commands, it looks much cleaner than plain text.)
        return f"$y = {s}$"

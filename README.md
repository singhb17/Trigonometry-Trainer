# Trig Sketching Trainer

A desktop study app for trig transformation practice (sine, cosine, tangent, cotangent, secant, cosecant).

## Features
- Two quiz modes:
  - **Equation -> Graph:** you sketch; use **Reveal Answer** to view the graph.
  - **Graph -> Equation:** you write the equation; use **Reveal Answer** to compare.
- Randomized parameters: amplitude, period/stretch, phase shift, vertical shift.
- Choose which trig functions appear (sin/cos/tan/cot/sec/csc).
- Multi-period view with anchor options.
- Optional equivalent-form hints for sin/cos.
- Dark/light theme toggle.

## Requirements
- Python 3.10+
- `numpy`, `matplotlib`

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Run
```bash
python src/main_frame.py
```

## Optional: Install as a package
```bash
pip install .
trig-trainer
```

## Notes
- This is a study tool; it does not automatically grade typed equations.
- Matplotlib uses the TkAgg backend to integrate with Tkinter.

## Troubleshooting
- If the graph window is blank, confirm `matplotlib` is installed in your active environment.
- If imports fail, double-check that you ran the app from this folder.

## License
Choose a license and add a `LICENSE` file (MIT, Apache-2.0, GPL-3.0, etc.).

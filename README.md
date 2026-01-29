# Trignometry Sketching/Equations Trainer

A desktop study app for practicing transformations of trignometric functions (sine, cosine, tangent, cotangent, secant, cosecant).

## Features
- Two quiz modes:
  - **Equation -> Graph:** you sketch; use **Reveal Answer** to view the graph.
  - **Graph -> Equation:** you write the equation; use **Reveal Answer** to compare.
- Randomized parameters: amplitude, period/stretch, phase shift, vertical shift.
- Choose which trig functions appear (sin/cos/tan/cot/sec/csc).
- Multi-period view with the option to anchor view at the origin.
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

## Troubleshooting
- If the graph window is blank, confirm `matplotlib` is installed in your active environment.
- If imports fail, double-check that you ran the app from this folder.

## Note from Developer
- This is a study tool; it does not automatically grade typed equations.
- Matplotlib uses the TkAgg backend to integrate with Tkinter.

- This project was developed with intentional use of AI-assisted programming tools. The goal was not code automation for speed, but exploration of how modern AI systems can be integrated into a real development workflow — including design decisions, refactoring, debugging, and modularization.
All architectural decisions, logic validation, and final structure were reviewed and refined by the developer. This project reflects an ongoing effort to understand the practical limits, strengths, and trade-offs of AI-supported software development while building a usable application.

## License
Choose a license and add a `LICENSE` file (MIT, Apache-2.0, GPL-3.0, etc.).

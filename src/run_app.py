"""
run_app.py
----------
Launcher for the Streamlit interface that automatically folds in
feedback and retrains models when you stop the app.

Run this INSTEAD of `streamlit run app.py` directly:

    python run_app.py

What happens:
  1. Starts the Streamlit app normally -- use it exactly as before.
  2. When you press Ctrl+C in THIS terminal, Streamlit stops, and this
     script automatically runs, in order:
        - src/incorporate_feedback.py  (folds any saved corrections in)
        - src/evaluate.py              (retrains all models on the result)

Uses sys.executable throughout (not a bare "streamlit"/"python" command)
so everything runs through the SAME Python your venv is using -- this
also sidesteps the Windows PATH issue where a bare `streamlit` command
can resolve to a different Python installation than your activated venv.
"""

import subprocess
import sys


def main():
    print("Starting Streamlit... press Ctrl+C here to stop and auto-retrain.\n")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        pass  # this IS the expected way to stop -- not an error

    print("\nStreamlit stopped. Folding in feedback and retraining...\n")
    subprocess.run([sys.executable, "src/incorporate_feedback.py"])
    subprocess.run([sys.executable, "src/evaluate.py"])
    print("\nDone -- models retrained on any feedback saved this session.")


if __name__ == "__main__":
    main()

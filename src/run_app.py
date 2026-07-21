import subprocess
import sys


def main():
    print("Starting Streamlit... press Ctrl+C here to stop and auto-retrain.\n")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        pass

    print("\nStreamlit stopped. Folding in feedback and retraining...\n")
    subprocess.run([sys.executable, "src/incorporate_feedback.py"])
    subprocess.run([sys.executable, "src/evaluate.py"])
    print("\nDone -- models retrained on any feedback saved this session.")


if __name__ == "__main__":
    main()

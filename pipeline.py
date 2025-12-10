import subprocess
import sys
from pathlib import Path

# Paths to scripts inside src/
PIPELINE_STEPS = [
    "src/etl.py",
    "src/train.py",
    "src/forecast.py"
]

def run_step(script_path):
    print(f"\n▶ Running: {script_path}\n" + "-"*50)

    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

    # Print script output
    print(result.stdout)
    if result.stderr:
        print("⚠️ ERRORS:")
        print(result.stderr)

    # Stop pipeline if a step fails
    if result.returncode != 0:
        print(f" Pipeline stopped! Error in: {script_path}")
        sys.exit(1)

def run_pipeline():
    print("\n Starting End-to-End ML Pipeline...\n")

    for step in PIPELINE_STEPS:
        run_step(step)

    print("\n** PIPELINE COMPLETED SUCCESSFULLY! **")
    print("All steps executed: ETL → Training → Forecasting\n")

if __name__ == "__main__":
    run_pipeline()

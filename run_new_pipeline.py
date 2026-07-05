import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PAPERMILL_KERNEL = os.environ.get("PAPERMILL_KERNEL", "fonda-venv")

# Define notebooks and outputs
jobs = [
    ("MLP-experience_replay_confidently_correct_memory.ipynb",
     "MLP-experience_replay_confidently_correct_memory_output.ipynb"),
    ("MLP-experience_replay_hard_example_mining.ipynb",
     "MLP-experience_replay_hard_example_mining_output.ipynb"),
    ("MLP-experience_replay_uncertainity_prioritization.ipynb",
     "MLP-experience_replay_uncertainity_prioritization_output.ipynb"),
]

# Run each notebook in sequence
for input_nb, output_nb in jobs:
    print(f"Running {input_nb} -> {output_nb}")
    subprocess.run(
        [sys.executable, "-m", "papermill", "--kernel", PAPERMILL_KERNEL, input_nb, output_nb],
        check=True,
        cwd=BASE_DIR,
    )

print("All notebooks executed successfully.")

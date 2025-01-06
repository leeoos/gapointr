import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "../results/images/")
os.makedirs(SAVE_DIR, exist_ok=True)

def get_args():
    parser = argparse.ArgumentParser(
        prog="Plot losses",
        description="""Plot Training Loss""",
    )
    parser.add_argument(
        "--data_file", type=str,
        help="Path to data file",
        required=True
    )
    parser.add_argument(
        "--title", type=str,
        help="Plot title",
        required=True
    )
    parser.add_argument(
        "--label", type=str,
        help="Plot label",
        required=True
    )
    return parser.parse_args()


def main():

    args = get_args()
    title = args.title
    label = args.label
    x_name = "Steps" if "steps" in title.lower() else "Epochs" 

    name = "_".join(title.split(" "))

    # Load the new datasets
    file_path = args.data_file

    with open(file_path, 'r') as file: 
        data = json.load(file)

        # Extract steps and flatten data
        losses = []
        for key, value in data.items():
            losses.append(np.mean(value))

        total_steps_new = len(losses)
        step_indices_new = range(1, total_steps_new + 1)

        # Plot the new data
        plt.figure(figsize=(12, 6))
        plt.plot(step_indices_new, losses, label=label, marker='o', markersize=2, linestyle='-', color='red')

        plt.title(title)
        plt.xlabel(x_name)
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{name}.png")) 

if __name__ == "__main__":
    main()
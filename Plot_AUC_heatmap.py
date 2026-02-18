import argparse
import os
import sys
import glob
from Function_ import draw_auc_heatmap

# Ensure current directory is in path for imports if needed
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(description="Generate AUC Heatmap from CSV")
parser.add_argument(
    "--models-dir",
    type=str,
    default=(
        "./models"
        if os.environ.get("MODEL_DIR") is None
        else os.environ.get("MODEL_DIR")
    ),
    help="Base models directory",
)
parser.add_argument(
    "--models-idx",
    type=str,
    default="260209_105821",  # 260206_084839, 260209_105821, 260210_081156
    help="Model timestamp/index folder (e.g. 260206_084839).",
)
parser.add_argument(
    "--search-pattern",
    type=str,
    default="AUC_summary*.csv",
    help="Pattern to search for CSV if exact path not provided. Default: AUC_summary*.csv",
)
parser.add_argument(
    "--csv-path",
    type=str,
    default=None,
    help="Direct path to the AUC summary CSV file. Overrides models-dir/models-idx logic.",
)

args = parser.parse_args()


def main():
    csv_file = args.csv_path

    if csv_file is None:
        # If no CSV path, must have models-idx
        if not args.models_idx:
            print("Error: Must provide either --csv-path or --models-idx")
            return

        # Search in the model folder
        search_dir = os.path.join(args.models_dir, args.models_idx)
        if not os.path.exists(search_dir):
            print(f"Error: Directory {search_dir} does not exist.")
            return

        files = glob.glob(os.path.join(search_dir, args.search_pattern))
        if not files:
            print(
                f"Error: No CSV file matching '{args.search_pattern}' found in {search_dir}"
            )
            return

        # Pick the one with largest size or most specific name?
        # Usually checking for case number case1, case4 etc.
        # Just pick the first one matching
        csv_file = files[0]
        print(f"Found CSV: {csv_file}")

    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} does not exist")
        return

    # Determine output path (same folder as csv, named similarly)
    output_path = csv_file.replace(".csv", "_heatmap.png")

    print(f"Generating heatmap for {csv_file}...")
    draw_auc_heatmap(csv_file, output_path, title_suffix="")
    print("Done.")


if __name__ == "__main__":
    main()

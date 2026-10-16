"""AUC 히트맵 스크립트: Test_multi_models.py가 저장한 AUC 표(CSV)를 색 칠한 표 그림(히트맵)으로 다시 그린다.

Test_multi_models.py도 실행이 끝나면 같은 그림을 그린다. 이 스크립트는 모델을 다시 평가하지 않고,
이미 있는 CSV만으로 그림을 다시 만들 때 쓴다(예: 그림 모양을 바꾼 뒤).

실행 예
    python Plot_AUC_heatmap.py --models-idx <RUN_ID>          # models/<RUN_ID>/에서 AUC_summary*.csv를 찾는다
    python Plot_AUC_heatmap.py --csv-path ./models/<RUN_ID>/AUC_summary_case1.csv

결과: CSV와 같은 폴더에 <CSV 이름>_heatmap.png (예: AUC_summary_case1_heatmap.png).
그리는 일은 mcdl/plotting.py의 draw_auc_heatmap이 한다.
"""

import argparse
import os
import sys
import glob
from mcdl.plotting import draw_auc_heatmap

# Ensure current directory is in path for imports if needed
sys.path.append(os.getcwd())

# 명령줄 옵션 정의. 실제 읽기(parse_args)는 main 안에서 한다 (import만 할 때는 아무것도 실행되지 않게).
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


def main(argv=None):
    """CSV 경로를 정하고(--csv-path, 없으면 모델 폴더에서 --search-pattern으로 찾은 첫 파일) 히트맵 PNG를 저장한다.

    argv를 주면 명령줄 대신 그 목록을 옵션으로 읽는다. 쓰는 곳: 이 파일 맨 아래 실행부, tests/test_scripts.py.
    """
    args = parser.parse_args(argv)
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

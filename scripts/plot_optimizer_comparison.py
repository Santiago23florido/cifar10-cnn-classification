import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path as _Path

from scripts.section4_visualization import DEFAULT_FIGURE_DIR, DEFAULT_RESULTS_PATH, save_optimizer_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=_Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--figure-dir", type=_Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    save_optimizer_figures(args.results, args.figure_dir)


if __name__ == "__main__":
    main()

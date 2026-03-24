# Section 4 plotting script reused by the notebook and report.
import argparse

from scripts.section_4 import DEFAULT_RESULTS_PATH
from scripts.section_4.visualization import DEFAULT_FIGURE_DIR, save_optimizer_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Section 4 optimizer figures.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--figure-dir", type=str, default=str(DEFAULT_FIGURE_DIR))
    args = parser.parse_args()
    save_optimizer_figures(args.input, args.figure_dir)


if __name__ == "__main__":
    main()

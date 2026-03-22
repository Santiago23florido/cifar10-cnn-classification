import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path as _Path

from scripts.section4_pipeline import DEFAULT_RESULTS_PATH, ensure_determinism, run_section4_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=_Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    ensure_determinism()
    run_section4_pipeline(output_path=args.output, force_recompute=args.force_recompute, verbose=True)


if __name__ == "__main__":
    main()

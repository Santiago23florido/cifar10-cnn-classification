# CLI entry point that mirrors the notebook Section 4 pipeline.
import argparse

from scripts.section_4 import DEFAULT_RESULTS_PATH, ensure_determinism, run_section4_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Section 4 experimental campaign.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    ensure_determinism()
    run_section4_pipeline(output_path=args.output, force_recompute=args.force_recompute, verbose=True)


if __name__ == "__main__":
    main()

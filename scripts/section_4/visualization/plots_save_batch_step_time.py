import matplotlib
import matplotlib.pyplot as plt
from ..runtime import Path, np

def save_batch_step_time(summary: list[dict[str, float]], figure_dir: Path) -> Path:
    batch_sizes = [row["batch_size"] for row in summary]
    means = [row["mean_step_time"] for row in summary]
    stds = [row["std_step_time"] for row in summary]

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(batch_sizes, means, yerr=stds, marker="o", capsize=4, linewidth=2)
    plt.xlabel("Batch size")
    plt.ylabel("Mean time per step (s)")
    plt.title("Batch size comparison: step time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = figure_dir / "section4_batch_step_time.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

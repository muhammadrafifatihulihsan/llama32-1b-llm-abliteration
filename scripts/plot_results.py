"""Plot the saved response metrics without loading a model."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    directory = Path(__file__).resolve().parents[1] / "results" / "reference"
    with (directory / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = {"baseline": "Baseline\nTransformerLens",
              "abliterated": "Abliterated\nHugging Face",
              "ollama_abliterated": "Abliterated\nOllama"}
    colors = ["#315a78", "#16857b", "#bd6c39"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True)
    fig.patch.set_facecolor("#f8fafc")
    for ax, metric, title in zip(axes, ["refusal_rate", "asr_heuristic"],
                                 ["Keyword refusal rate", "ASR heuristic*"]):
        values = [float(row[metric]) for row in rows]
        bars = ax.bar(range(len(rows)), values, color=colors, width=0.58)
        ax.set_xticks(range(len(rows)), [labels[row["condition"]] for row in rows])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title, loc="left", fontsize=13, pad=15)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                    f"{value:.2%}", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle("Llama 3.2 1B · Response comparison", fontsize=18,
                 x=0.075, ha="left", y=0.98)
    fig.text(0.075, 0.91, "80 harmful prompts per condition · Saved run results",
             color="#475569", fontsize=11)
    fig.text(0.075, 0.055,
             "*Non-refusal responses longer than 50 characters; not validated harmful compliance.\n"
             "Prompt sets differ and extraction/evaluation overlap exists. See docs/results.md.",
             fontsize=9, color="#475569", linespacing=1.6)
    fig.subplots_adjust(top=0.79, bottom=0.22, left=0.075, right=0.98, wspace=0.15)
    destination = directory / "metrics.png"
    fig.savefig(destination, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {destination.relative_to(directory.parents[1])}")


if __name__ == "__main__":
    main()

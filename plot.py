import numpy as np
import matplotlib.pyplot as plt
from split import load_eurosat_paths


def plot_side_by_side(
    simple_acc,
    strong_acc,
    num_classes,
):
    epochs_simple = simple_acc.shape[0]
    epochs_strong = strong_acc.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    _, _, class_to_idx = load_eurosat_paths("EuroSAT_RGB")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # ----- Simple augmentation -----
    for c in range(num_classes):
        axes[0].plot(
            range(1, epochs_simple + 1),
            simple_acc[:, c],
            label=idx_to_class[c]
        )

    axes[0].set_title("Simple Augmentation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Per-class Accuracy / TPR")
    axes[0].grid(True)

    print(idx_to_class)
    # ----- Strong augmentation -----
    for c in range(num_classes):
        axes[1].plot(
            range(1, epochs_strong + 1),
            strong_acc[:, c],
            label=idx_to_class[c]
        )

    axes[1].set_title("Strong Augmentation")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)

    # Shared legend


    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.15, 0.5)
    )

    plt.suptitle("Per-class Accuracy Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("eurosat_per_class_simple_vs_strong.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    num_classes = 10
    simple_per_class_acc = np.load("results/per_class_acc_simple.npy")
    strong_per_class_acc = np.load("results/per_class_acc_strong.npy")
    print(simple_per_class_acc.mean())
    print(strong_per_class_acc.mean())
    assert 0
    plot_side_by_side(
        simple_per_class_acc,
        strong_per_class_acc,
        num_classes=num_classes
    )

if __name__ == "__main__":
    main()

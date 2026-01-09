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
    for c in range(num_classes):
        axes[1].plot(
            range(1, epochs_strong + 1),
            strong_acc[:, c],
            label=idx_to_class[c]
        )

    axes[1].set_title("Strong Augmentation")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)

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


def plot_ms_per_class(ms_acc, num_classes):
    epochs = ms_acc.shape[0]

    _, _, class_to_idx = load_eurosat_paths("EuroSAT_RGB")  # using RGB names
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    plt.figure(figsize=(10, 6))

    for c in range(num_classes):
        plt.plot(
            range(1, epochs + 1),
            ms_acc[:, c],
            label=idx_to_class[c]
        )

    plt.title("MS Model Per-Class Accuracy Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Per-Class Accuracy / TPR")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/per_class_acc_ms6.png", dpi=300)
    plt.show()


def main():
    num_classes = 10
    dataset = "ms"

    if dataset == "rgb":
        simple_per_class_acc = np.load("results/per_class_acc_simple.npy")
        strong_per_class_acc = np.load("results/per_class_acc_strong.npy")
        print("Mean accuracy per epoch:", simple_per_class_acc.mean(axis=1))
        print("Mean accuracy per epoch strong:", strong_per_class_acc.mean(axis=1))
        print(simple_per_class_acc.mean())
        print(strong_per_class_acc.mean())
        plot_side_by_side(
            simple_per_class_acc,
            strong_per_class_acc,
            num_classes=num_classes
        )
    elif dataset == "ms":
        ms_per_class_acc = np.load("results/per_class_acc_ms6.npy")
        print("Mean accuracy per epoch:", ms_per_class_acc.mean(axis=1))
        print(ms_per_class_acc.mean())
        plot_ms_per_class(ms_per_class_acc, num_classes=num_classes)

if __name__ == "__main__":
    main()

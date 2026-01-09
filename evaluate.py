import random
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from train import EuroSATDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

seed = 3732848
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if torch.backends.mps.is_available():
    torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def show_images(entries, title):
    plt.figure(figsize=(15, 3))
    for i, item in enumerate(entries):
        img = Image.open(item["path"])
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{item['score']:.2f}")
    plt.suptitle(title)
    plt.show()

def run_inference(save_path=None):
    num_classes = 10
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load("eurosat_simple.pth"))
    model = model.to(device)
    model.eval()


    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = EuroSATDataset(
        split_file="splits/test.txt",
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    image_paths = [p for p, _ in test_dataset.samples]
    if save_path:
        torch.save(
            {
                "logits" :all_logits,
                "labels": all_labels,
                "paths": image_paths,
            },
            save_path,
        )
    return all_logits, all_labels


def reproduction_pipeline(new_logits_path, save=False, show_img=False):
    """
    Perform the entire inference process on the test
    set using the saved model. If the `save` option is set,
    the logits will be saved and overwrite the old ones.

    For reproduction, we set this to `False` in order
    to compare the inference results to the saved logits.

    The `show_img` flag simply shows the top-5 for the
    three classes, which was mainly used during debugging
    and can be ignored when attempting result reproduction.
    """

    if new_logits_path:
        saved = torch.load(new_logits_path, weights_only=False)
        all_logits = saved["logits"]
        all_labels = saved["labels"]
        image_paths = saved["paths"]
    else:
        all_logits, all_labels, image_paths = run_inference()

    selected_classes = [0, 3, 7]

    assert all_logits.shape[0] == len(image_paths)

    results = {}

    for c in selected_classes:
        class_scores = all_logits[:, c].numpy()

        sorted_idx = np.argsort(class_scores)

        bottom_5_idx = sorted_idx[:5]
        top_5_idx = sorted_idx[-5:][::-1]

        results[c] = {
            "top_5": [
                {
                    "path": image_paths[i],
                    "score": class_scores[i],
                    "true_label": all_labels[i],
                }
                for i in top_5_idx
            ],
            "bottom_5": [
                {
                    "path": image_paths[i],
                    "score": class_scores[i],
                    "true_label": all_labels[i],
                }
                for i in bottom_5_idx
            ],
        }

    for c in selected_classes:
        print(f"\n=== Class {c} ===")

        print("Top-5 scoring images:")
        for item in results[c]["top_5"]:
            print(
                f"{item['path']} | "
                f"score={item['score']:.3f} | "
                f"true_label={item['true_label']}"
            )

        print("\nBottom-5 scoring images:")
        for item in results[c]["bottom_5"]:
            print(
                f"{item['path']} | "
                f"score={item['score']:.3f} | "
                f"true_label={item['true_label']}"
            )

    # TODO: Use the idx2class dict here for better clarity ?
    for c in selected_classes:
        show_images(results[c]["top_5"], f"Class {c} - Top 5")
        show_images(results[c]["bottom_5"], f"Class {c} - Bottom 5")

    if save:
        torch.save(
            {
                "logits" :all_logits,
                "labels": all_labels,
                "paths": image_paths,
            },
            "results/test_logits_simple.pt",
        )
    else:
        saved = torch.load("results/test_logits_simple.pt", weights_only=False)
        saved_logits = saved["logits"]
        saved_labels = saved["labels"]
        saved_paths = saved["paths"]

        assert image_paths == saved_paths
        assert all_logits.shape == saved_logits.shape
        assert torch.equal(all_labels, saved_labels)

        diff = torch.max(torch.abs(all_logits - saved_logits))
        print(
            f"Max absolute difference between computed and saved logits: {diff:.6f}"
        )


def main():
    new_logits_path = "results/your_logits.pt"
    run_inference(new_logits_path)

    # This will compare 
    reproduction_pipeline(new_logits_path)


if __name__ == "__main__":
    main()

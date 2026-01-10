import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter



def load_eurosat_paths(dataset_path):
    class_names = sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith(".")
    )

    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    image_paths = []
    labels = []

    for cls_name in class_names:
        cls_dir = os.path.join(dataset_path, cls_name)
        cls_idx = class_to_idx[cls_name]

        for fname in os.listdir(cls_dir):
            if fname.startswith("."):
                continue
            fpath = os.path.join(cls_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(
                (".jpg", ".jpeg", ".png", ".tif", ".tiff")
            ):
                image_paths.append(fpath)
                labels.append(cls_idx)

    return np.array(image_paths), np.array(labels), class_to_idx


def create_splits(
    dataset_path="EuroSAT_RGB",
    train_size=2500,
    val_size=1000,
    test_size=2000,
    seed=3732848,
):
    # Load data
    image_paths, labels, class_to_idx = load_eurosat_paths(dataset_path)

    total_needed = train_size + val_size + test_size
    assert total_needed <= len(image_paths), "Requested split sizes exceed dataset size"

    # Step 1: select only the required total subset
    X_subset, _, y_subset, _ = train_test_split(
        image_paths,
        labels,
        train_size=total_needed,
        stratify=labels,
        random_state=seed,
    )

    # Step 2: split train
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_subset,
        y_subset,
        train_size=train_size,
        stratify=y_subset,
        random_state=seed,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=test_size,
        stratify=y_tmp,
        random_state=seed,
    )

    assert len(set(X_train) & set(X_val)) == 0
    assert len(set(X_train) & set(X_test)) == 0
    assert len(set(X_val) & set(X_test)) == 0

    print("Total images:", len(image_paths))
    print("Class mapping:", class_to_idx)
    print("Train:", Counter(y_train))
    print("Val:", Counter(y_val))
    print("Test:", Counter(y_test))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_splits_to_txt(split, save_dir, filename):
    X, y = split
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "w") as f:
        for path, label in zip(X, y):
            f.write(f"{path} {label}\n")

    print(f"Saved {filename} ({len(X)} samples)")


def rgb_splits():
    train_split, val_split, test_split = create_splits(
        dataset_path="EuroSAT_RGB",
        train_size=2500,
        val_size=1000,
        test_size=2000,
    )

    project_root = os.path.abspath(".")
    splits_dir = os.path.join(project_root, "splits")

    save_splits_to_txt(train_split, splits_dir, "train.txt")
    save_splits_to_txt(val_split, splits_dir, "val.txt")
    save_splits_to_txt(test_split, splits_dir, "test.txt")


def ms_splits():
    train_split, val_split, test_split = create_splits(
        dataset_path="EuroSAT_MS",
        train_size=2500,
        val_size=1000,
        test_size=2000,
    )

    project_root = os.path.abspath(".")
    splits_dir = os.path.join(project_root, "splits")

    save_splits_to_txt(train_split, splits_dir, "train_ms.txt")
    save_splits_to_txt(val_split, splits_dir, "val_ms.txt")
    save_splits_to_txt(test_split, splits_dir, "test_ms.txt")


def main():
    # rgb_splits()
    ms_splits()


if __name__ == "__main__":
    main()

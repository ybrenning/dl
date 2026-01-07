import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

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

    return image_paths, labels, class_to_idx


def create_splits():
    seed = 3732848

    dataset_path = "EuroSAT_RGB"
    # dataset_path = "EuroSAT_MS"

    image_paths, labels, class_to_idx = load_eurosat_paths(dataset_path)
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    print(len(image_paths))
    print(class_to_idx)

    from sklearn.model_selection import train_test_split

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=2000, stratify=labels, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=1000, stratify=y_train_val, random_state=42
    )
    train_set = set(X_train)
    val_set = set(X_val)
    test_set = set(X_test)

    assert len(train_set & val_set) == 0, "Train and Val sets overlap!"
    assert len(train_set & test_set) == 0, "Train and Test sets overlap!"
    assert len(val_set & test_set) == 0, "Val and Test sets overlap!"
    from collections import Counter

    print("Train:", Counter(y_train))
    print("Val:", Counter(y_val))
    print("Test:", Counter(y_test))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_splits_to_txt(split, save_dir, filename):
    X, y = split
    os.makedirs(save_dir, exist_ok=True)  # create subdirectory if missing
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w") as f:
        for path, label in zip(X, y):
            f.write(f"{path} {label}\n")
    print(f"Saved {filename} to {save_dir}")

def main():
    train_split, val_split, test_split = create_splits()

    # IDK how else this is supposed to be set
    project_root = "."
    project_root = os.path.abspath(project_root)

    splits_subdir = "splits"
    splits_dir = os.path.join(os.path.abspath(project_root), splits_subdir)


    save_splits_to_txt(train_split, splits_dir, "train.txt")
    save_splits_to_txt(val_split, splits_dir, "val.txt")
    save_splits_to_txt(test_split, splits_dir, "test.txt")


if __name__ == "__main__":
    main()

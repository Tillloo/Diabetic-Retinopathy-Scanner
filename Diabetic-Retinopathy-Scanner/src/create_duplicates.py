import os
import pandas as pd
import random
import shutil


def main():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")

    LABELS_FILE = os.path.join(DATA_DIR, "trainLabels.csv")

    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Directory not found: {TRAIN_DIR}")

    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")

    labels = pd.read_csv(LABELS_FILE)

    fnames_no_extension = [
        "".join(file.split(".")[:-1]) for file in os.listdir(TRAIN_DIR)
    ]
    filtered_labels = labels[labels["image"].isin(fnames_no_extension)]

    counts = filtered_labels["level"].value_counts().to_dict()
    max_count = max(counts.values())

    file_counts = {}

    class_to_images = filtered_labels.groupby("level")["image"].apply(list).to_dict()

    for i in range(5):
        if i not in counts:
            raise ValueError("Need at least 1 of each class")

        for _ in range(counts[i], max_count):
            file_to_duplicate = random.choice(class_to_images[i])

            count = file_counts.get(file_to_duplicate, 1) + 1

            duplicate_file_name = f"{file_to_duplicate}-{count}"
            shutil.copy(
                os.path.join(TRAIN_DIR, f"{file_to_duplicate}.jpeg"),
                os.path.join(TRAIN_DIR, f"{duplicate_file_name}.jpeg"),
            )
            filtered_labels.loc[len(filtered_labels)] = [duplicate_file_name, i]

            file_counts[file_to_duplicate] = count

    filtered_labels.to_csv(os.path.join(DATA_DIR, "new_trainLabels.csv"), index=False)


if __name__ == "__main__":
    main()

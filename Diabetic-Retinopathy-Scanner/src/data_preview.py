import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt


def get_dataset(dataset_type="train"):
    if dataset_type not in {"train", "test"}:
        raise ValueError(f"{dataset_type} is not valid")

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")

    if dataset_type == "train":
        DATASET_DIR = TRAIN_DIR
        # LABELS_FILE = os.path.join(DATA_DIR, "trainLabels.csv")
    else:
        DATASET_DIR = TEST_DIR
        # LABELS_FILE = os.path.join(DATA_DIR, "testLabels.csv")

    LABELS_FILE = os.path.join(DATA_DIR, "trainLabels.csv")

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"{dataset_type.capitalize()} directory not found: {DATASET_DIR}"
        )

    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")

    labels = pd.read_csv(LABELS_FILE, dtype=str)
    labels["image"] = labels["image"] + ".jpeg"

    fnames = os.listdir(DATASET_DIR)
    mask = labels["image"].isin(fnames)
    filtered_labels = labels[mask].sort_values("image")

    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32

    def clahe_single_image(img):
        img = np.array(img)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 1:
            img = cv2.merge([img.squeeze()] * 3)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        l = l.astype(np.uint8)
        a = a.astype(np.uint8)
        b = b.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        if l.shape != a.shape or l.shape != b.shape:
            a = cv2.resize(a, (l.shape[1], l.shape[0]))
            b = cv2.resize(b, (l.shape[1], l.shape[0]))

        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def apply_clahe_tf(images):
        def process_image(img):
            img = tf.numpy_function(clahe_single_image, [img], tf.uint8)
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
            return img

        return tf.map_fn(process_image, images, fn_output_signature=tf.float32)

    def enhance_vessels_tf(images_grey):
        sobel = tf.image.sobel_edges(images_grey)
        x = sobel[:, :, :, :, 0]
        y = sobel[:, :, :, :, 1]
        magnitude = tf.sqrt(tf.square(x) + tf.square(y))

        return tf.image.convert_image_dtype(magnitude, tf.float32)

    def remove_background_tf(images, images_grey):
        def remove_single_image(image, image_grey):
            blur = tf.image.adjust_contrast(image_grey, 2)
            binary_mask = tf.cast(blur > tf.reduce_mean(blur), tf.float32)

            return image * tf.concat([binary_mask] * 3, axis=-1)

        return tf.map_fn(
            lambda pair: remove_single_image(pair[0], pair[1]),
            (images, images_grey),
            fn_output_signature=tf.float32,
        )

    def preprocess_image_tf(images):
        images_grey = tf.image.rgb_to_grayscale(images)
        images = apply_clahe_tf(images)
        images = remove_background_tf(images, images_grey)
        images = enhance_vessels_tf(images_grey)
        return images

    dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True if dataset_type == "train" else False,
        seed=123,
        labels=filtered_labels["level"].astype(int).to_list(),
        label_mode="int",
        color_mode="rgb",
    )

    dataset = dataset.map(lambda x, y: (preprocess_image_tf(x), y))

    if dataset_type == "train":
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.2),
                layers.RandomBrightness(factor=0.1),
                layers.RandomContrast(factor=0.2),
            ]
        )
        dataset = dataset.map(lambda x, y: (data_augmentation(x), y))

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and visualize dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "test"],
        help='Type of dataset to load: "train" or "test". Default is "train".',
    )
    args = parser.parse_args()

    dataset = get_dataset(dataset_type=args.dataset)

    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(labels[i].numpy())
            plt.axis("off")

    plt.tight_layout()
    plt.show()

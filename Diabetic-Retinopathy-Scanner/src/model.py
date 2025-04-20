import tensorflow as tf
from keras import layers, models
from keras.applications import EfficientNetB0

from data_preview import get_dataset


def build_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    train_dataset = get_dataset("train")
    test_dataset = get_dataset("test")

    model = build_model()

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
        ],
    )

    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

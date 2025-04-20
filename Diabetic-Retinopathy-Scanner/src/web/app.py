import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, url_for
from threading import Timer
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)

MODEL_PATH = "resNet_50.keras"

model = tf.keras.models.load_model(MODEL_PATH)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        if file and allowed_file(file.filename):
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            try:
                img_array = preprocess_image(file_path)

                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=-1)[0]

                return render_template(
                    "result.html",
                    file_url=url_for("static", filename=file.filename),
                    predicted_class=predicted_class,
                    probabilities=predictions[0].tolist(),
                )
            except Exception as e:
                return f"Error processing file: {e}", 500
            finally:
                Timer(
                    60,
                    lambda: os.remove(file_path) if os.path.exists(file_path) else None,
                ).start()

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

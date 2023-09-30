from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__, static_url_path='/static')

model = tf.keras.models.load_model("model_dis.hdf5")
class_names = ['Bean_angular_leaf_spot', 'Bean_bean_rust', 'Rice_Bacterial leaf blight', 'Rice_Brown_Spot', 'Rice_Healthy', 'Rice_Hispa', 'Rice_Leaf smut', 'Rice_Leaf_Blast', 'Wheat_Brown_Rust', 'Wheat_Healthy', 'Wheat_Yellow_Rust']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")
        
        file = request.files["file"]
        
        if file.filename == "":
            return render_template("index.html", error="No selected file")
        
        if file:
            image = Image.open(file)
            resized_image = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(resized_image)
            img_array = np.expand_dims(img_array, 0)
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = round(100 * np.max(predictions[0]), 2)
            return render_template("index.html", prediction=predicted_class, confidence=confidence)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

from fastapi import FastAPI, File, UploadFile
from tensorflow import keras
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.utils import img_to_array
from PIL import Image
import numpy as np
import json
import io

app = FastAPI()

model_path = "model/fish2eat_model.h5"
labels_path = "model/labels.json"

model = load_model(model_path)

with open(labels_path, "r") as file:
    class_labels = json.load(file)

@app.get("/")
def home():
    return {"message": "Welcome to the Fish2Eat Recognition API"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        result = {
            "class": class_labels[str(predicted_class)],
            "confidence": f"{confidence * 100:.2f}%",
        }
        return result
    except Exception as e:
        return {"error": str(e)}

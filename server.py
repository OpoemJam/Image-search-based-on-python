import os
import sys

import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime

from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

from pathlib import Path
from util import base64_to_pil

from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/your_model.h5'

# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path, encoding='bytes', allow_pickle=True))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))   #destinated image path and feature path
features = np.array(features)


@app.route('/search', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)# save the uploaded image in the ‘static/uploaded/’ folder

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]# output the top30 images and print their distance

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

@app.route('/crop', methods=['GET', 'POST'])
def crop():
    return render_template('index2.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('index3.html')
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None

if __name__=="__main__":
    app.run("0.0.0.0")
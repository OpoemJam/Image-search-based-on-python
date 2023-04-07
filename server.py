import os
import sys
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime

from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

from pathlib import Path


app = Flask(__name__)

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
        
        describe=("黑腿信天翁是一种凶猛的海鸟")
        return render_template('index.html',
                               query_describe=describe,
                               query_path=uploaded_img_path,
                               scores=scores)
 
    else:
        return render_template('index.html')

@app.route('/crop', methods=['GET', 'POST'])
def crop():
    return render_template('index2.html')

@app.route('/text', methods=['GET', 'POST'])
def TBIR():
    return render_template('index3.html')

if __name__=="__main__":
    app.run("0.0.0.0")
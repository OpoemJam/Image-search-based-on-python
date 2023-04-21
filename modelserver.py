import numpy as np
from PIL import Image
from VGG16feature_extractor import VGG16FeatureExtractor
from VGG19feature_extractor import VGG19FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Read image features
VGG16fe = VGG16FeatureExtractor()
VGG19fe = VGG19FeatureExtractor()
VGG16features = []
VGG19features = []
VGG16img_paths = []
VGG19img_paths = []

for VGG16feature_path in Path("./static/VGG16feature").glob("*.npy"):
    VGG16features.append(np.load(VGG16feature_path, encoding='bytes', allow_pickle=True))
    VGG16img_paths.append(Path("./static/img") / (VGG16feature_path.stem + ".jpg"))
VGG16features = np.array(VGG16features)

for VGG19feature_path in Path("./static/VGG19feature").glob("*.npy"):
    VGG19features.append(np.load(VGG19feature_path, encoding='bytes', allow_pickle=True))
    VGG19img_paths.append(Path("./static/img") / (VGG19feature_path.stem + ".jpg"))
VGG19features = np.array(VGG19features)

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('main.html')

@app.route('/VGG16', methods=['GET', 'POST'])
def VGG16():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + "VGG16" +datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)# save the uploaded image in the ‘static/uploaded/’ folder

        # Run search
        query = VGG16fe.extract(img)
        dists = np.linalg.norm(VGG16features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], VGG16img_paths[id]) for id in ids]# output the top30 images and print their distance
        
        VGG16describe="你正在使用VGG16模型"
        
        return render_template('model.html',
                               modeldescribe=VGG16describe,
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

@app.route('/VGG19', methods=['GET', 'POST'])
def VGG19():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + "VGG19" +datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)# save the uploaded image in the ‘static/uploaded/’ folder

        # Run search
        query = VGG19fe.extract(img)
        dists = np.linalg.norm(VGG19features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], VGG19img_paths[id]) for id in ids]# output the top30 images and print their distance
        
        VGG19describe="你正在使用VGG19模型"
        
        return render_template('model.html',
                               modeldescribe=VGG19describe,
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0")

import os
import numpy as np
from VGG16feature_extractor import VGG16FeatureExtractor
from VGG19feature_extractor import VGG19FeatureExtractor
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

from flask import Flask,render_template,url_for,send_from_directory,request,session,redirect
from flask_avatars import Avatars

from pathlib import Path

import base64

app=Flask(__name__)
avatars=Avatars(app)

app.secret_key='19062314'
app.config['AVATARS_SAVE_PATH']='labeled/'

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

@app.route('/avatars/<path:filename>')
def get_avatar(filename):
    return send_from_directory(app.config['AVATARS_SAVE_PATH'], filename)

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
        
        desc=scores[0]
        textdesc=''.join(str(i) for i in desc)
        textdesc2=textdesc.split('\\')[2].rsplit('_', 2)[-3]
        with open("describe/"+textdesc2 + ".txt","r", encoding='utf-8') as file:
            describe=file.read()
        
        VGG16describe="你正在使用VGG16模型"
        
        return render_template('index.html',
                               model_describe=VGG16describe,
                               query_describe=describe,
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
        
        desc=scores[0]
        textdesc=''.join(str(i) for i in desc)
        textdesc2=textdesc.split('\\')[2].rsplit('_', 2)[-3]
        with open("describe/"+textdesc2 + ".txt","r", encoding='utf-8') as file:
            describe=file.read()
        
        VGG19describe="你正在使用VGG19模型"
        
        return render_template('index.html',
                               model_describe=VGG19describe,
                               query_describe=describe,
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

@app.route('/crop', methods=['GET', 'POST'])
def crop():
    return render_template('index2.html')

@app.route('/label1', methods=['GET', 'POST'])
def label1():
    if request.method == 'POST':
        file1 = file2 = request.files.get('file')
        raw_filename = avatars.save_avatar(file1)
        session['raw_filename'] = raw_filename
        
        upload_path=("./labeled/uploaded.jpg")
        img=Image.open(file2)
        img.save(upload_path)
        return redirect(url_for('label2'))
    return render_template('upload.html')

@app.route('/label2', methods=['GET', 'POST'])
def label2():
    if request.method == 'POST':
        x = request.form.get('x')
        y = request.form.get('y')
        w = request.form.get('w')
        h = request.form.get('h')
        
        x1=float(x)
        y1=float(y)
        w1=float(w)
        h1=float(h)
        
        img=Image.open("./labeled/uploaded.jpg")
        width, height = img.size
        
        pic3 = pic1 = Image.new('RGB', (width,height), 'black')
        
        draw = ImageDraw.Draw(pic1)
        draw.rectangle((x1, y1, x1+w1, y1+h1), outline="red", fill='red')
        pic1.save('./labeled/background.jpg')
        
        draw1 = ImageDraw.Draw(pic3)
        draw1.rectangle((x1, y1, x1+w1, y1+h1), outline="red", fill='red')
        draw1.rectangle(( width-160 , height-80, width , height ),fill='white')
        draw1.rectangle(( width-140 , height-70 , width-120 , height-50 ), fill='black')
        draw1.rectangle(( width-140 , height-30 , width-120 , height-10 ), fill='red')
        
        font=ImageFont.truetype('arial.ttf',15)
        text_color=(0,0,0)
        draw1.text((width-115,height-70),'1 background', font=font, fill=text_color)
        draw1.text((width-115,height-30),'2 objective', font=font, fill=text_color)
        
        pic2=Image.open('./labeled/uploaded.jpg')
        miximg=Image.blend(pic3, pic2, 0.45)
        miximg.save('./labeled/miximg.jpg')
        
        with open('./labeled/uploaded.jpg','rb') as img_file:
            data1 = img_file.read()
            imgdata1 = base64.b64encode(data1).decode("utf-8")
            
        with open('./labeled/background.jpg','rb') as img_file:
            data2 = img_file.read()
            imgdata2 = base64.b64encode(data2).decode("utf-8")
            
        with open('./labeled/miximg.jpg','rb') as img_file:
            data3 = img_file.read()
            imgdata3 = base64.b64encode(data3).decode("utf-8")
            
        return render_template('done.html', img_data1=imgdata1, img_data2=imgdata2, img_data3=imgdata3 )
    return render_template('crop.html')

@app.route('/upload', methods=['GET', 'POST'])
def database():
    if request.method == 'POST':
        file = request.files['addtodatabaseimg']
        img = Image.open(file.stream)
        uploaded_database_img = datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save("static/img/"+uploaded_database_img)
        img.save("static/VGG16feature/"+uploaded_database_img)
        img.save("static/VGG19feature/"+uploaded_database_img)
        
        dir_path=["static/VGG16feature/"]
        
        old_form=".jpg"
        for filename in os.listdir(dir_path):
            if filename.endswith(old_form):
                new_filename = filename[:-len(old_form)] + ".npy"
                os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, new_filename))
        
        dir_path=["static/VGG19feature/"]
        
        for filename in os.listdir(dir_path):
            if filename.endswith(old_form):
                new_filename = filename[:-len(old_form)] + ".npy"
                os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, new_filename))   
        
        uploadsuccess="上传成功！但注意如果您想要使用新的图片数据，需要重新启动服务器并增加图片相关描述文件，并按格式来修改图片文件及其特征文件名称。"
        return render_template('index4.html',
                               query_describe=uploadsuccess)
    else:
        return render_template('index4.html')

if __name__=="__main__":
    app.run("0.0.0.0", debug=True)
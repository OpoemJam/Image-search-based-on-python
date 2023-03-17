# 基于python的图像搜索引擎/Image-search-based-on-python

本项目是基于内容的图像搜索（CBIR）引擎，旨在明确理解搜索引擎各部分的功能，明确各部分之间的参数交换，并且将引擎各部分进行模块化，使得各部分可以被替换，作为一个可持续发展的项目。

This project is a content-based image search（CBIR）engine, the purpose is to clearly understand each parts' functions, clarify the parameter exchange between each part, and modularize each part of the engine so that each part can be replaced , it's aimed to be a redevelopable project.

# 脚本功能介绍/description of scripts functions
feature_extractor.py:

图像处理函数，使用VGG-16体系结构和来自ImageNet的预先训练的权重。该函数的功能是预处理图像，因为vgg-16对输入图片有一定要求。提取图像特征存入对应的.npy文件中以备后续使用。

Image processing script, use VGG-16architecture and pretrained weights from ImageNet.The function of this script is to preprocess images, because VGG-16 has certain requirements for input images. Then it will extract image features and store them in the corresponding .npy files for future use.

offline.py:

使用先前构建的图像处理函数提取本地数据库中图像特征储存在对应的.npy文件中。

Use the image processing script built before to extract local database images' features and store them in the corresponding .npy files.

server.py:

使用flask库构建简易的网络框架，将上传的图像存放到‘/static/uploaded/’中，提取图像特征，使用‘np.linalg.norm’函数与数据库图像进行计算相似度，返回相似度最高的三十张图像。

Use flask to bulid simple network frame, store the uploaded images in‘/static/uploaded/’ folder, use ‘np.linalg.norm’ to calculate its feature's similarity with the datebase images, then return 30 the most similar images.

# 使用方法/How to launch
1.在‘/static/img/'中放入要作为数据库的.jpg图片。

2.在‘/static/feature’中放入相应的.npy文件。

3.运行offline.py。

4.运行server.py。

5.进入提示框返回的网址即可使用图像搜索。

1.put .jpg picture in ‘/static/img/' that needed to be database.

2.put corresponding .npy files in '/static/feature/'.

3.run offline.py

4.run server.py

5.go to the website that returned from Ipython console and use the engine.

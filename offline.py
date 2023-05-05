from PIL import Image
from VGG16feature_extractor import VGG16FeatureExtractor
from VGG19feature_extractor import VGG19FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    VGG19fe = VGG19FeatureExtractor()
    VGG16fe = VGG16FeatureExtractor()
    
    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = VGG16fe.extract(img=Image.open(img_path))
        VGG16feature_path = Path("./static/VGG16feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(VGG16feature_path, feature)
        feature = VGG19fe.extract(img=Image.open(img_path))
        VGG19feature_path = Path("./static/VGG19feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(VGG19feature_path, feature)
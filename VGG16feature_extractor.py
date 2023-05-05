from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os

# See https://keras.io/api/applications/ for details
'''
IF YOU DONT HAVE GPU OR YOU JUST WANT TO USE CPU, DELTE '#' BEFORE OS.ENVIRON.
IT'S STRONGLY RECOMMENDED TO USE GPU ACCLERATION.
'''
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class VGG16FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')  #Use VGG-16 as the architecture and ImageNet for the weight
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output) #Customize the model to return features from fully-connected layer

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize
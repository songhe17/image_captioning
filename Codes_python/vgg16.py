import numpy as np
import tensorflow as tf
import keras
import os,os.path
import _pickle as pickle
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
class encoder(object):

    def __init__(self, image_path,save_path):
        self.path = image_path
        self.save_path = save_path
    def _extract_feats(self):
        model = VGG16(include_top = False)
        path = self.path
        data = {}
        for filename in os.listdir(path):
            image = load_img(os.path.join(path,filename),target_size = (224,224))
            image = img_to_array(image)
            image = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))
            image = preprocess_input(image)
            labels = model.predict(image)
            labels = np.reshape(labels,(49,512))
            filename = filename.split('.')[0]
            data[filename] = labels
        with open(self.save_path,'wb') as f:
            pickle.dump(data,f)

#trythisone = encoder('/Users/songhewang/Desktop/test_images/','/Users/songhewang/Desktop/try.pkl')
#trythisone._extract_feats()
#with open('/Users/songhewang/Desktop/try.pkl','rb') as f:
#    data = pickle.load(f)
#    print(np.shape(data['000000000001']))

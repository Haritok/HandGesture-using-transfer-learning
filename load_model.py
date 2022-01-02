
import os
import keras
import tensorflow as tf
import matplotlib.style as style
import numpy as np
import tables
from PIL import Image
from keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models
import time
from keras.models import load_model
from matplotlib import pyplot as plt

style.use('seaborn-whitegrid')

"""### Open images and convert to NumPy arrays"""

image = image_utils.load_img(path='.../gesture_data/00/01_palm/frame_00_01_0001.png', target_size=(224, 224))
image = image_utils.img_to_array(image)

print(image.shape)

lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('../gesture_data/00'):
    if not j.startswith('.'):
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1




def get_data(start, stop):
    x_data = []
    y_data = []
    datacount = 0 # We'll use this to tally how many images are in our dataset
    for i in range(start, stop): # Loop over the ten top-level folders
        for j in os.listdir('../gesture_data/0' + str(i) + '/'):
            if not j.startswith('.'): # Again avoid hidden folders
                count = 0 # To tally images of a given gesture
                for k in os.listdir('../gesture_data/0' + 
                                    str(i) + '/' + j + '/'):
                                    # Loop over the images
                    img = Image.open('../gesture_data/0' + 
                                     str(i) + '/' + j + '/' + k).convert('L')
                                    # Read in and convert to greyscale
                    img = img.resize((224, 224)).convert('L')
                    arr = np.array(img)
                    x_data.append(arr) 
                    count = count + 1
                y_values = np.full((count, 1), lookup[j]) 
                y_data.append(y_values)
                datacount = datacount + count
                
    return x_data, y_data

def process_data(x_data, y_data):
    x_data = np.array(x_data, dtype = 'float32')
    x_data = np.stack((x_data,) * 3, axis=-1)
    # x_data = np.array(x_data, dtype=np.uint8)
    print(len(x_data))
    print(x_data.shape)
    x_data = x_data.reshape(len(x_data),224, 224, 3)
    x_data /= 255
    
    
    y_data = np.array(y_data)
    y_data = y_data.reshape(len(x_data), 1)
    y_data = to_categorical(y_data)
    
    return x_data, y_data

"""### Train-Test Split

Train-test split - totally separating images from the first 8 people, and the last 2 people
"""

X_train, y_train = get_data(0,8)

#calculating time taken to process the training data
start = time.time()

X_train, y_train = process_data(X_train, y_train)

end = time.time()



#X_train = preprocess_input(X_train) 
X_test, y_test = get_data(8,10)

#calculating time taken to process the training data
start = time.time()

X_test, y_test = process_data(X_test, y_test)


#X_test = preprocess_input(X_test) 

#print("type of x_data-->",type(x_data))
#converting 
#x_data = np.array(x_data, dtype = 'float32')

#

# load model
model = load_model('VGG-16_model.h5')

model.summary()


#pred = model.predict(X_test)
#print("pred shape===",pred)
#pred = np.argmax(pred, axis=1)
#y_true = np.argmax(y_test, axis=1)
#print(confusion_matrix(y_true, pred))
#print(classification_report(y_true, pred))












#



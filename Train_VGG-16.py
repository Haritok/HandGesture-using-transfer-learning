
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
from matplotlib import pyplot as plt

style.use('seaborn-whitegrid')

image = image_utils.load_img(path='../gesture_data/00/01_palm/frame_00_01_0001.png', target_size=(224, 224))
image = image_utils.img_to_array(image)

print(image.shape)

lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('.../gesture_data/00'):
    if not j.startswith('.'):
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1




def get_data(start, stop):
    x_data = []
    y_data = []
    datacount = 0 # We'll use this to tally how many images are in our dataset
    for i in range(start, stop): # Loop over the ten top-level folders
        for j in os.listdir('.../gesture_data/0' + str(i) + '/'):
            if not j.startswith('.'): # Again avoid hidden folders
                count = 0 # To tally images of a given gesture
                for k in os.listdir('.../gesture_data/0' + 
                                    str(i) + '/' + j + '/'):
                                    # Loop over the images
                    img = Image.open('.../gesture_data/0' + 
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

## Train-Test Split


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


#

# Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(
    weights='imagenet',
    include_top=False,
)

model_vgg16_conv.summary()

# Create your own input format (here 224x224x3)
img_input = Input(shape=(224, 224, 3), name='image_input')

# makes the layers non-trainable
for layer in model_vgg16_conv.layers:
    layer.trainable = False
    
# Use the generated model 
output_vgg16_conv = model_vgg16_conv(img_input)


# Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
# x = Dense(4096, activation='relu', name='fc1')(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(10, activation='softmax', name='predictions')(x)  # here the 2 indicates binary (3 or more is multiclass)

# Create your own model 
my_model = Model(img_input,x)

# In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()

model1 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#optimizer1 = optimizers.Adam() # Adam is like a gradient descent (way to find parameters)

base_model = model1  # Topless
# Add top layer
x = base_model.output
x = Flatten()(x)
x = Dropout(0.5)(x)

predictions = Dense(10, activation='softmax')(x)
model = Model(base_model.input,predictions)

# Train top layer
for layer in base_model.layers:
    layer.trainable = False
    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test),verbose=1)

#Evaluation metrics for VGG 16
#predict_x=model.predict(X_test) 
#classes_x=np.argmax(predict_x,axis=1)
#predictions = (model.predict(X_test) > 0.5).astype("int32")
#from sklearn.metrics import accuracy_score
#a = accuracy_score(y_test,predictions)
#print(a)

model.save('.../VGG-16_model.h5')
          

#pred = model.predict(X_test)
#pred = np.argmax(pred, axis=1)
#y_true = np.argmax(y_test, axis=1)
#print(confusion_matrix(y_true, pred))
#print(classification_report(y_true, pred))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()







#



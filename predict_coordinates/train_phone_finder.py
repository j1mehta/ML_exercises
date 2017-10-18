
# coding: utf-8

import glob
import cv2
import math
import os
import sys
import numpy as np
import imutils
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD
from keras.models import Sequential
from keras.models import Model 
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D
from keras.preprocessing.image import array_to_img, img_to_array, load_img

epochs = 20
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True, clipvalue=0.5)

def get_im(path):
    '''loads image as PIL
    Returns
        A 3D Numpy array
    Numpy array x has format (height, width, channel)
    '''
    img = load_img(path)
    size = (224,224)
    img = img.resize(size) 
    
    return img_to_array(img)

def load_train(labels, folder):
    X_train = []
    y_train = labels
    print('Read train images')
    for index, row in labels.iterrows():
        fileName =  str(folder) +row['imgID']
        img = get_im(fileName)
        X_train.append(img)
        y_train.append
    X_train, X_test, y_train, y_test = split_validation_set(X_train, y_train, 0.2)
    y_train = np.array(zip(y_train.x, y_train.y))
    y_test = np.array(zip(y_test.x, y_test.y))
    X_train = np.array(X_train)/255
    X_test = np.array(X_test)/255    
    return X_train, X_test, y_train, y_test

def rotate(px, py, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = (112,112)

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)

#Create Horizontal and Vertical Flips
def augmentation(labels, rot60=False, rot90=False, rot30=False, flipH=True, flipV=True):
    row_horiz = []
    row_vert = []
    row_R30 = []
    row_R45 = []
    row_R60 = []
    row_R90 = []
    cols = ['imgID', 'x', 'y']
    for index, row in labels.iterrows():
        fileName =  str(folder) +row['imgID']
        img = get_im(fileName)

        if(rot30):
            imgR30 = imutils.rotate(img, -30)
            fileNameR30 =  str(folder) +'R30'+row['imgID']
            cv2.imwrite(fileNameR30, imgR30)
            qx30, qy30 = rotate(int(row['x']*224), int(row['y']*224), 0.523599)
            row_R30.append(['R30'+row['imgID'], qx30, qy30])    

        if(rot60):
            imgR60 = imutils.rotate(img, -60)
            fileNameR60 =  str(folder) +'R60'+row['imgID']
            cv2.imwrite(fileNameR60, imgR60)
            qx60, qy60 = rotate(int(row['x']*224), int(row['y']*224), 1.0472)
            row_R60.append(['R60'+row['imgID'], qx60, qy60]) 

        if(rot90):
            imgR90 = imutils.rotate(img, -90)
            fileNameR90 =  str(folder) + 'R90'+row['imgID']
            cv2.imwrite(fileNameR90, imgR90)
            qx90, qy90 = rotate(int(row['x']*224), int(row['y']*224), 1.5708)
            row_R90.append(['R90'+row['imgID'], qx90, qy90]) 

        if(flipH):
            imgFH = cv2.flip(img,1)
            fileNameH = str(folder) +'HF'+row['imgID']
            cv2.imwrite(fileNameH, imgFH)
            row_horiz.append(['HF'+row['imgID'], 1-row['x'], row['y']])

        if(flipV):
            imgFV = cv2.flip(img,0)
            fileNameV =  str(folder) +'VF'+row['imgID']
            cv2.imwrite(fileNameV, imgFV)
            row_vert.append(['VF'+row['imgID'], row['x'], 1-row['y']])


    dfH = pd.DataFrame(row_horiz, columns=cols)
    dfV = pd.DataFrame(row_vert, columns=cols)
    dfR30 = pd.DataFrame(row_R30, columns=cols)
    dfR60 = pd.DataFrame(row_R60, columns=cols)
    dfR90 = pd.DataFrame(row_R90, columns=cols)

    labels = labels.append(dfH,ignore_index=True)
    labels = labels.append(dfV,ignore_index=True)
    labels = labels.append(dfR30,ignore_index=True)
    labels = labels.append(dfR60,ignore_index=True)
    labels = labels.append(dfR90,ignore_index=True)

    return labels
    
def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def CNN(weights_path=None):
    input_shape=(224, 224, 3)
    model = Sequential()
    model.add(Conv2D(20, (4, 4), input_shape = input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear'))
    if weights_path:
        model.load_weights(weights_path)
    return model

def transfer_learning_vgg(weights_path=None):
    #load vgg model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_model.layers[:]:
        layer.trainable = False
    print('Model loaded.')

    #initialise top model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(2, activation='linear'))

    # add the model on top of the convolutional base
    model = Model(input= vgg_model.input, output= top_model(vgg_model.output))
    if(weights_path):
        model.load_weights(weights_path)
    return model

def training(X_train, y_train, transfer=False):
    if(transfer):        
        model = transfer_learning_vgg()
    else:
        model = CNN()
    model.compile(loss='mse', optimizer=sgd)
    model.fit(X_train, y_train, batch_size=16, epochs=epochs, verbose=1, shuffle = True)
    model.save_weights('weights.h5')
    
def display_result():
    count = 0
    for index, row in y_test.iterrows():
        fileName =  str(folder) +row['imgID']
        img = get_im(fileName)
        img = cv2.circle(img, (int(224*res[count][0]),  int(res[count][1]*224)), 10, (255,0,0))
        img = Image.fromarray(img.astype('uint8'))
        img.show()
        count +=1
        
def predict():    
    res = model.predict(X_test)
    print(mean_squared_error(res, y_test))
    
if __name__ == '__main__':
    folder = sys.argv[1]
    print(folder)
    names = ['imgID', 'x', 'y']
    labels = pd.read_csv(str(folder) + '/labels.txt', names=names, delim_whitespace=True)
    labels = augmentation(labels, flipH=False, flipV=False)
    X_train, X_test, y_train, y_test = load_train(labels, folder)
    training(X_train, y_train)

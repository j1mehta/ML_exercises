# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 15:40:55 2017

@author: j.mehta
"""
#%%
import pandas as pd
from sklearn import cross_validation
from keras.models import Sequential
from keras.layers import Dense
#%%

#%%
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['Pregnant', 'Gluco_Concen', 'BP', 'TricepThickness', 'Insulin', 'BMI', 'Pedigree', 'Age','class']
#Forming panda dataframe
df = pd.read_csv(url, names = names)
#%%
#Preparing training and validation set
#Numpy representation of NDFrame
numpyArr = df.values
X = numpyArr[:,0:8]
Y = numpyArr[:,8]

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#%%
#Create and compile the model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#%%
#Training and evaluation
model.fit(X_train, Y_train, nb_epoch=150, batch_size=10)
#%%
scores = model.evaluate(X_validation, Y_validation)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 11:28:27 2017

@author: j.mehta
"""
#%%
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
#%%
#Keras downloads images to ./keras/datasets/
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#%%
#The subplot() command specifies numrows, numcols, fignum where fignum ranges from 1 to numrows*numcols. The commas in the subplot command are optional if numrows*numcols<10. So subplot(211) is identical to subplot(2, 1, 1). You can create an arbitrary number of subplots and axes
#3,3 is the dimension of the final plot. 1 is the figure no. on that final plot. For more clarity, plot 
#multiple images by changing 3X3 to 2X2. It will increase each figure's size since it can accomodate 4
#images (1 to 2X2) instead of 9 (1 to 3X3)
#%%
plt.subplot(331)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(332)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(333)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
#%%
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
#%%
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
#numpy_array.reshape(no_of_rows, no_of_columns) float32 fixes precision
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
#%%
#Normalization: WHY (http://cs231n.github.io/neural-networks-1/#actfun)
#Sigmoids saturate and kill gradients. A very undesirable property of the sigmoid neuron is that when the neuron’s activation saturates at either tail of 0 or 1, the gradient at these regions is almost zero. Recall that during backpropagation, this (local) gradient will be multiplied to the gradient of this gate’s output for the whole objective. Therefore, if the local gradient is very small, it will effectively “kill” the gradient and almost no signal will flow through the neuron to its weights and recursively to its data. Additionally, one must pay extra caution when initializing the weights of sigmoid neurons to prevent saturation. For example, if the initial weights are too large then most neurons would become saturated and the network will barely learn.
#Thus, never use sigmoid as activation function, use Relu instead. I don't think normalization is required if using Relu
# normalize inputs from 0-255 to 0-1

#With Normalization: Error = 1.81%
#Without Normalization: Error = 52.64%
X_train = X_train / 255
X_test = X_test / 255
#%%
# one hot encode outputs
#When One Hot encoding is used on a particular data set (a matrix) and used as training data for learning algorithms, it gives significantly better results with respect to prediction accuracy, compared to using the original matrix itself as training data.
#Why?
#Many learning algorithms either learn a single weight per feature, or they use distances between samples. The former is the case for linear models such as logistic regression, which are easy to explain.
#Suppose you have a dataset having only a single categorical feature "nationality", with values "UK", "French" and "US". Assume, without loss of generality, that these are encoded as 0, 1 and 2. You then have a weight w for this feature in a linear classifier, which will make some kind of decision based on the constraint w×x + b > 0, or equivalently w×x < b.
#The problem now is that the weight w cannot encode a three-way choice. The three possible values of w×x are 0, w and 2×w. Either these three all lead to the same decision (they're all < b or ≥b) or "UK" and "French" lead to the same decision, or "French" and "US" give the same decision. There's no possibility for the model to learn that "UK" and "US" should be given the same label, with "French" the odd one out.
#By one-hot encoding, you effectively blow up the feature space to three features, which will each get their own weights, so the decision function is now w[UK]x[UK] + w[FR]x[FR] + w[US]x[US] < b, where all the x's are booleans. In this space, such a linear function can express any sum/disjunction of the possibilities (e.g. "UK or US", which might be a predictor for someone speaking English).
#Similarly, any learner based on standard distance metrics (such as k-nearest neighbors) between samples will get confused without one-hot encoding. With the naive encoding and Euclidean distance, the distance between French and US is 1. The distance between US and UK is 2. But with the one-hot encoding, the pairwise distances between [1, 0, 0], [0, 1, 0] and [0, 0, 1] are all equal to √2.
#This is not true for all learning algorithms; decision trees and derived models such as random forests, if deep enough, can handle categorical variables without one-hot encoding.
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)
num_classes = y_test.shape[1]
#%%
# define baseline model
#Softmax(multinomial logistic regression) explained here: https://www.quora.com/What-is-the-intuition-behind-SoftMax-function
#Optimizer: Although we used "Adam" here, it is important to note the original: SGD. https://www.quora.com/Whats-the-difference-between-gradient-descent-and-stochastic-gradient-descent
#Generally speaking, instead of taking a large step towards the steepest slope on a hill, in stochastic gradient descent you take small steps, which happens to be much faster in practice. You don't have to sum over all the training examples to compute the gradient, you compute the gradient for each training example or small batches at a time.

#No point in using second order optimization methods (like Newton's method) since the are impractical because of the computationally 
#expensive Hessian matrix. But, optimization methods in which learning rate adapts itself are the stte of the art. Among such methods
#are Adagrad, RMSprop and Adam in which Adam works best since it doesn't use the raw gradient for parameter update, rather a smoothened 
#one. 
#Nesterov momentum is also used if learning rate is to be kept constant for all paramter updates unlike Adam. I received an error of 7.91%
#as compared to Adam which gave an error of 1.81%.
#
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
	model.add(Dense(num_classes, init='normal', activation='softmax'))
	# Compile model
	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
     	#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 #%%
 # build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
#%%

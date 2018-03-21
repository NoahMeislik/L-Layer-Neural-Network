import numpy as np
import time
import h5py
from helper_functions import *
from utils import *
import scipy
from scipy import ndimage
import pickle

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

parameters = load_obj("parameters")
num_px = parameters['num_px']
print(num_px)

#pred_test = predict(test_x, test_y, parameters)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)


classes = ["non-cat", "cat"]


my_image = "download (1).jpg" #  
my_label_y = [1] # 

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

prediction = "Cat" if 1 in np.squeeze(my_predicted_image) else "Non Cat"
print(prediction)
print("y = " + str(np.squeeze(my_predicted_image)) + ", the model predicts a \"" + prediction +  "\" picture.")


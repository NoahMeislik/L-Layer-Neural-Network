
import numpy as np
import time
import h5py
from helper_functions import *
import scipy
from scipy import ndimage
#tickle my pickle
import pickle

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

parameters = load_obj("parameters")
num_px = parameters['num_px']

my_image = "download (1).jpg" #  
my_label_y = [1] # 

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

prediction = "Cat" if 1 in np.squeeze(my_predicted_image) else "Non Cat"
print("y = " + str(np.squeeze(my_predicted_image)) + ", the model predicts a \"" + prediction +  "\" picture.")


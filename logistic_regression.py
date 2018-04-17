import numpy as np
import time
import h5py
import pickle

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:]) 
    train_y = np.array(train_dataset["train_set_y"][:]) 
    
    train_y = train_y.reshape((1, train_y.shape[0]))
    
    return train_x, train_y


def initialize(dimensions):
    """
    Initializes parameters for logistic regression
    """
    return np.zeros(shape=(dimensions, 1)), 0


def sigmoid(Z):
    """
    Performs the sigmoid function.
    """
    return 1 / (1+np.exp(-Z))

def cost(A, Y, x_shape):
    return (- 1 / x_shape)*np.sum(Y*np.log(A) + (1 - Y)*(np.log(1 - A)))

def forward_propagation(X, Y, w, b):
    x_shape = X.shape[1]

    A = sigmoid(X)

    A = np.dot(w.T, X)+b
    A = sigmoid(A)

    return np.squeeze(cost(A, Y, x_shape))

def backward_propagation(A, X, Y):
    x_shape = X.shape[1]

    dw = (1 / x_shape) * np.dot(X,(A-Y).T)
    db = (1 / x_shape) * np.sum(A-Y)

    return dw, db

def gradient_descent(X, Y, w, b, iterations, alpha):
    for i in range(iterations):
        A = forward_propagation(X, Y, w, b)
        dw, db = backward_propagation(A, X, Y)

        w -= alpha * dw
        b -= alpha * db

    return w, b, dw, db

def predict(X, w, b):
    x_shape = X.shape[1]
    Y = np.zeros((1,x_shape))

    w = w.reshape(X.shape[0], 1)

    A = np.dot(w.T, X)+b
    A = sigmoid(A)

    for i in range(A.shape[1]):
        prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return prediction

Xtrain, Ytrain = load_data()

m_train = Ytrain.shape[1]
num_px = Xtrain.shape[1]

Xtrain = Xtrain.reshape(Xtrain.shape[0], -1).T
train_set_x = Xtrain / 255.


def model(Xtrain, Ytrain, iterations=4000, alpha=0.4):

    w, b = initialize(Xtrain.shape[0])

    w, b, dw, db = gradient_descent(Xtrain, Ytrain, w, b, iterations, alpha)

    prediction = predict(Xtrain, w, b)
    accuracy = 100-np.mean(np.abs(prediction - Ytrain) * 100)
    print("The accuracy of the model is: " + accuracy)

    parameters = {
        "w": w,
        "b": b,
        "alpha": alpha,
        "iterations": iterations,
        "accuracy": accuracy
    }

    return parameters

parameters = model(Xtrain, Ytrain)
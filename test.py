import numpy as np
import h5py

def load_train_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:]) 
    train_y = np.array(train_dataset["train_set_y"][:]) 
    
    train_y = train_y.reshape((1, train_y.shape[0]))
    
    return train_x, train_y

x_train, y_train = load_train_data()




x_train = x_train.reshape(x_train.shape[0], -1).T
print("Reshaped data of size: " + str(x_train.shape[0]))

x_train = x_train / 255.

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))   

def initialize(dimensions):   
    return np.zeros(shape=(dimensions, 1)), 0

def cost(A, Y, x_shape):
    return (- 1 / x_shape)*np.sum(Y*np.log(A) + (1 - Y)*(np.log(1 - A)))

def forward_propagation(w, b, X, Y):
    
    x_shape= X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)  

    return np.squeeze(cost(A, Y, x_shape)), A

def backward_propagation(w, b, X, A, Y):
    
    x_shape = X.shape[1]
    dw = (1 / x_shape) * np.dot(X, (A - Y).T)
    db = (1 / x_shape) * np.sum(A - Y)
    
    return dw, db

def gradient_descent(w, b, X, Y, num_iterations, alpha):
    
    costs = []
    
    for i in range(num_iterations):
        

        cost, A = forward_propagation(w, b, X, Y)
        
        dw, db = backward_propagation(w, b, X, A, Y)
        
        w -= alpha * dw 
        b -= alpha * db
        
        if i % 100 == 0:
            costs.append(cost)
            print ("The cost after {} steps is: {}".format(i, cost))
        
    
    return w, b, dw, db, costs

def predict(w, b, X):

    
    x_shape = X.shape[1]
    prediction  = np.zeros((1, x_shape))


    w = w.reshape(X.shape[0], 1)
    A = np.dot(w.T, X)+b
    A = sigmoid(A)

    
    for i in range(A.shape[1]):

        A[0, i] = 1 if A[0,i] > 0.5 else 0
        
    return A

def test_set_accuracy(X_test, Y_test, parameters):
    test_prediction = predict(paramaters["w"}, parameters["b"], X_test)

    accuracy = 100 - np.mean(np.abs(test_prediction - Y_test)) * 100)

    return accuracy



def model(x_train, y_train, iterations=2000, alpha=0.5, print_cost=False):

    w, b = initialize(x_train.shape[0])


    w, b, dw, db, costs = gradient_descent(w, b, x_train, y_train, iterations, alpha)
    
   
    train_prediction = predict(w, b, x_train)

    accuracy = 100 - np.mean(np.abs(train_prediction - y_train)) * 100)



    print("Model finished, the accuracy of the training set is: {}".format(accuracy)
    
    parameters = {"w": w,
         "b" : b, 
         "costs" : costs,
         "alpha" : alpha,
         "iterations": iterations}
    
    return parameters

paramaters = model(x_train, y_train, iterations = 2000, alpha = 0.005)

print("The test set had an accuracy of: {}".format(test_set_accuracy(x_test, y_test, paramaters)))

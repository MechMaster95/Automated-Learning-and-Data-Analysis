## HW3-Q5 Sample Solution
# Environment: Python 3.5, Keras 2.0.6, Theano 0.9.0 backend, 
##
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import pylab

def setmodel(input_dim, hiddim):
    model = Sequential() # Define a model
    model.add(Dense(hiddim, input_dim=input_dim,activation='relu')) # Set a hidden layer
    model.add(Dense(1, activation='sigmoid')) # Set an output layer
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # Compile the model
    return model

def nnlearning(train_scores, val_scores, hiddim, input_dim, models):
    # Set a model
    model = setmodel(input_dim, hiddim)
    # Fit the model
    model.fit(xtrain, ytrain, epochs=10, batch_size=50) 
    models.append(model)
    # evaluate the model
    scores = model.evaluate(xtrain, ytrain)
    train_scores.append(scores[1])
    print("\nTrain Acc: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    scores = model.evaluate(xval, yval)
    val_scores.append(scores[1])
    print("\nValidation Acc: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    scores = model.evaluate(xtest, ytest)
    print("\nTest Acc: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    return models, train_scores, val_scores

def drawgraph(x, y1, y2): 
    pylab.plot(x, y1, '-b', label='training')
    pylab.plot(x, y1, 'bo')
    pylab.plot(x, y2, '-r', label='validation')
    pylab.plot(x, y2, 'ro')
    pylab.legend(loc='upper left')
    pylab.ylim(0.6, 1.0)
    pylab.show()


# Load dataset
xtrain = np.genfromtxt('hw3q4/X_train.csv', delimiter=',')
xval = np.genfromtxt('hw3q4/X_val.csv', delimiter=',')
xtest = np.genfromtxt('hw3q4/X_test.csv', delimiter=',')
ytrain = np.genfromtxt('hw3q4/Y_train.csv', delimiter=',')
yval = np.genfromtxt('hw3q4/Y_val.csv', delimiter=',')
ytest = np.genfromtxt('hw3q4/Y_test.csv', delimiter=',')

print("X train: "+str(xtrain.shape))
# Init the parameters
input_dim = np.size(xtrain, 1)
hiddim = [2,4,6, 8, 10] 
train_scores = []
val_scores = []
models = []
np.random.seed(7)

# Training & Validataion
for h in hiddim:
    print('* hidden neurons: {0}'.format(h))
    models, train_scores, val_scores = nnlearning(train_scores,val_scores, h, input_dim, models)

# Get the optimal num of hidden neurons from the validation results
max_hid_idx = val_scores.index(max(val_scores))
print("\nOptimal number of hidden neurons:", hiddim[max_hid_idx])
# Test the data with the optimal num of hidden neurons
test_scores = models[max_hid_idx].evaluate(xtest, ytest)
print("\nTest:  %.2f%%" % ( test_scores[1]*100))

# Plot the training & validation accuracy 
drawgraph(hiddim, train_scores, val_scores)
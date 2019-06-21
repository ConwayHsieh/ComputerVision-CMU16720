import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    bound = np.sqrt(6) / np.sqrt( in_size + out_size)
    
    W = np.random.uniform(-bound, bound, (in_size, out_size))
    b = np.zeros(out_size)
    #W, b = None, None
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    #res = None
    res = 1/(1+np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    #pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    pre_act = np.dot(X, W) + b
    post_act = activation(pre_act)


    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    #res = None
    c = -np.amax(x, axis=1).reshape(-1,1)
    #print(c.shape)
    #print(x.shape)
    res = np.exp(x + c)
    total = np.sum(res, axis=1).reshape(-1,1)

    return res / total

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    #loss, acc = None, None
    loss = -np.sum(np.multiply(y, np.log(probs)))

    numExamples = y.shape[0]
    numCorr = np.zeros(numExamples)
    for i in range(numExamples):
        # find index of maximum probability class
        classIdx = np.argmax(probs[i,:])
        # since one-hot vector, only need to check for the single one
        if (y[i, classIdx] == 1):
            numCorr[i] = 1

    numCorr = np.sum(numCorr)
    acc = numCorr/numExamples
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X

    # do derivative via activation, multiply by errors
    der = activation_deriv(post_act) * delta

    # calculate derivatives
    grad_W = np.dot(np.transpose(X), der)
    grad_b = np.dot(np.ones((1,delta.shape[0])), der).flatten()
    grad_X = np.dot(der, np.transpose(W))

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    #initialize
    batches = []
    numData = x.shape[0]
    numBatch = numData/batch_size

    #randomize data order
    randOrder = np.random.permutation(numData)

    # split randomly ordered indices
    # into batches of equal sizes of size batch_size
    rand_batch_idx = np.array_split(randOrder, numBatch)

    # iterate through all batches
    for i in rand_batch_idx:
        batches.append((x[i], y[i]))

    return batches

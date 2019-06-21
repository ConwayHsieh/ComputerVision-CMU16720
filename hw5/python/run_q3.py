import numpy as np
import scipy.io, pickle, string
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']


max_iters = 50
# pick a batch size, learning rate
batch_size = 25
learning_rate = 1.5e-2
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

train_loss_list = []
train_acc_list =  []
valid_loss_list = []
valid_acc_list =  []

# initialize layers here
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, 36, params, 'output')


# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        #pass
        # training loop can be exactly the same as q2!
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs -yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        params['Wlayer1'] -= learning_rate*params['grad_Wlayer1']
        params['blayer1'] -= learning_rate*params['grad_blayer1']
        params['Woutput'] -= learning_rate*params['grad_Woutput']
        params['boutput'] -= learning_rate*params['grad_boutput']
    
    total_acc /= batch_num
    total_loss /= batch_num*batch_size

    # save training acc and loss for plotting
    train_acc_list.append(total_acc)
    train_loss_list.append(total_loss)

    # validation per iteration
    valid_h1 = forward(valid_x, params, 'layer1')
    valid_probs = forward(valid_h1, params, 'output', softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, valid_probs)

    # save valid acc and loss for plotting
    valid_acc_list.append(valid_acc)
    valid_loss_list.append(valid_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

# run on validation set and report accuracy! should be above 75%
valid_acc = None
valid_h1 = forward(valid_x, params, 'layer1')
valid_probs = forward(valid_h1, params, 'output', softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, valid_probs)

print('Validation accuracy: ',valid_acc)

# plot accuracy
epoch = np.arange(max_iters)
plt.figure('Accuracy')
fig = plt.gca()
fig.set_xlabel('Epoch')
fig.set_ylabel('Accuracy')
fig.plot(epoch, train_acc_list, 'r')
fig.plot(epoch, valid_acc_list, 'g')
fig.legend(['Training', 'Validation'])
plt.show()

# plot loss
plt.figure('Loss')
fig = plt.gca()
fig.set_xlabel('Epoch')
fig.set_ylabel('Cross-Entropy Loss')
fig.plot(epoch, train_loss_list, 'r')
plt.show()



import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

with open('q3_weights.pickle', 'rb') as handle:
    saved_params = pickle.load(handle)


weights_layer1 = saved_params['Wlayer1']
weights_init = params['Wlayer1']

fig = plt.figure('Visualize Layer 1 Weights - After')
grid = ImageGrid(fig, 111, nrows_ncols=(8,8))

for i in range(64):
    grid[i].imshow(weights_layer1[:,i].reshape(32,32))

fig = plt.figure('Visualize Layer 1 Weights - Initialization')
grid = ImageGrid(fig, 111, nrows_ncols=(8,8))

for i in range(64):
    grid[i].imshow(weights_init[:,i].reshape(32,32))


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()

# Q3.1.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# forward pass on validation dataset for confusion matrix
h1 = forward(valid_x, saved_params, 'layer1')
probs = forward(h1, saved_params, 'output', softmax)

# find the index = class of each prediction and corresponding truth
valid_true = np.argmax(valid_y, axis=1)
valid_pred = np.argmax(probs, axis=1)

# add x,y to confusion matrix, which is pred vs truth
for i in range(valid_true.shape[0]):
    confusion_matrix[valid_true[i], valid_pred[i]] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
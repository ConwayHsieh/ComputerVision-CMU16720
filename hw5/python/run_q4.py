import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(np.invert(bw), cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        #print(minr, minc)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    #boxList = []
    #for bbox in bboxes:
        # Y, X
        #minr, minc = bbox[0:2]
        #boxList.append((minr, minc))
    #    boxList.append(bbox)

    boxList = np.asarray(bboxes)
    #print(boxList)
    #print('\n')
    #boxList = boxList[np.argsort(boxList[:,1])]
    #boxList = np.split(boxList, np.where(np.diff(boxList[0,:]) > 20)[0]+1)
    
    # first column of y
    # split into arrays within the same row
    # ie when y values differ too much from previous y value
    dif = boxList[1:, 0] - boxList[:-1, 0]
    fi = np.where(abs(dif) > 50)[0]
    boxList = np.split(boxList, fi+1)
    #print(boxList)
    #print(type(boxList))
    #print(len(boxList))
    #print('\n')
    #boxList = boxList[np.argsort(boxList[:,0])]
    
    # sort each row by x values to get each row left to right on picture
    sortedBoxList = np.zeros(4, dtype=int)
    for i in range(len(boxList)):
        currRow = boxList[i]
        #print(currRow)
        sortedBoxList = np.vstack((sortedBoxList,
            currRow[np.argsort(currRow[:,1])]))
        #print(type(boxList[i]))
        #print(boxList[i], '\n')

    sortedBoxList = np.delete(sortedBoxList, (0), axis=0)
    #print(sortedBoxList)
    #print(type(boxList))
    #print('\n')
    """
    for bbox in sortedBoxList:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    """

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    lettersArray = []
    for bbox in sortedBoxList:
        minr, minc, maxr, maxc = bbox
        letter = bw[minr:maxr, minc:maxc]
        
        if '01' in img:
            pad = (25,25)
            letterPad = np.pad(
                letter, (pad, pad), 'constant', constant_values=0.0)
        else:   
            pad = (40,40)
            letterPad = np.pad(
                letter,(pad, pad), 'constant', constant_values=0.0)
        
        letterCrop = skimage.transform.resize(letterPad,(32,32))
        letterDil = skimage.morphology.dilation(letterCrop,skimage.morphology.square(2))

        #plt.imshow(letterCrop, cmap='gray')
        #plt.show()
        letterFlat = np.transpose(1.0 - letterDil).flatten()
        lettersArray.append(letterFlat)

    lettersArray = np.asarray(lettersArray)


    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    
    h1 = forward(lettersArray, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)

    predClass = np.argmax(probs, axis=1)
    #print(predClass)
    predLetters = letters[predClass]

    #print(np.array2string(predLetters))
    for l in predLetters:
        print(l, end='')

    print('\n')
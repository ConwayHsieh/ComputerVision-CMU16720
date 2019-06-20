import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import matplotlib
import skimage.io
import multiprocessing

#import pprint
#pp = pprint.PrettyPrinter(indent=4)

#debugging
import pdb
#pdb.set_trace()

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''



    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----
    filePathList = train_data['files']
    data_labels = train_data['labels']
    numImages = np.shape(filePathList)[0]
    dict_size = dictionary.shape[0]
    layer_num = 3
    vector_length = int(dict_size*(4**layer_num - 1)/3)
    features = np.zeros((numImages,vector_length))#, dtype='float64')
    args_list = []
    for i in range(numImages):
        currImgPath = "../data/" + filePathList[i]
        #currFeature = get_image_feature(currImgPath,dictionary,
        #    layer_num,dict_size,i)
        #features = np.vstack((features,currFeature))
        args_list.append((currImgPath,dictionary,layer_num,dict_size,i))

    pool = multiprocessing.Pool(processes = num_workers)
    pool.map(multiprocess_getFeature,args_list)
        
    for i in range(numImages):
        currFeature = np.load("y" + str(i) + ".npy")
        features[i,:] = currFeature
    #pdb.set_trace()
    #features = np.delete(features,(0),axis=0)
    save_filename = "trained_system"
    np.savez(save_filename,dictionary=dictionary, features=features,
        labels=data_labels, SPM_layer_num=layer_num)

    pass

def multiprocess_getFeature(args):
    currImgPath,dictionary,layer_num,dict_size,i = args
    currFeature = get_image_feature(currImgPath,dictionary,
        layer_num,dict_size)
    save_filename="y"+str(i)
    np.save(save_filename, currFeature)
    pass

def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    # ----- TODO -----
    # load data into vars
    testFilePaths = test_data['files']
    testLabels = test_data['labels']
    dictionary = trained_system['dictionary']
    trainFeatures = trained_system['features']
    trainLabels = trained_system['labels']
    layer_num = int(trained_system['SPM_layer_num'])
    dict_size = dictionary.shape[0]

    #
    numImages = np.shape(testFilePaths)[0]
    testPredLabelList = np.zeros(numImages,dtype=int)
    args_list = []
    for i in range(numImages):
        currTestFilePath = "../data/" + testFilePaths[i]
        #currImageFeature = get_image_feature(currTestFilePath,
        #    dictionary,layer_num,dict_size)
        args_list.append((currTestFilePath,dictionary,layer_num,dict_size,i))

    pool = multiprocessing.Pool(processes = num_workers)
    pool.map(multiprocess_getFeature,args_list)
    
    for i in range(numImages):
        currImageFeature = np.load("y" + str(i) + ".npy")
        distSet = distance_to_set(currImageFeature,trainFeatures)
        predLabel = trainLabels[np.argmax(distSet)]
        testPredLabelList[i] = predLabel

    #print(testPredLabelList)

    conf = np.zeros(8*8,dtype=int).reshape(8,8)
    for i in range(numImages):
        currPredLabel = testPredLabelList[i]
        truePredLabel = testLabels[i]
        conf[truePredLabel,currPredLabel] += 1

    accuracy = np.diag(conf).sum()/conf.sum()
    return conf, accuracy

def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    #load image
    image = skimage.io.imread(file_path)
    image = image.astype('float')/255
    # extract wordmap from image
    wordmap = visual_words.get_visual_words(image,dictionary)
    #compute SPM
    feature = get_feature_from_wordmap_SPM(wordmap,layer_num,K)
    #return computed feature
    return feature



def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    diff = np.minimum(word_hist,histograms)
    sim = np.sum(diff,axis=1)
    return sim


def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    # ----- TODO -----
    #print(dict_size)
    hist,bins = np.histogram(wordmap, bins=dict_size, density=True)
    #print(hist)
    #matplotlib.pyplot.hist(hist,bins=dict_size, density=True)
    #matplotlib.pyplot.bar(np.arange(len(hist)),hist)
    #matplotlib.pyplot.show()
    return hist



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    
    # ----- TODO -----

    L = layer_num - 1
    weights = []
    for l in range(layer_num):
        #print(l)
        if l == 0 or l == 1:
            weights.append(2**(-L))
        else:
            weights.append(2**(l-L-1))

    weights = weights[::-1]

    #print(weights)
    #print(np.hsplit(wordmap,4))
    #print(wordmap.shape)
    #
    Ls = np.arange(layer_num)[::-1]
    hist_all = np.zeros(1)
    for l in range(layer_num):
        currL = 2**Ls[l]
        #print(l)
        #print("meow1")
        #print(currL)
        currImgSplit = [A for sub in np.array_split(
            wordmap, currL, axis=0) for A in np.array_split(sub, currL, axis=1)]

        temp_hist = np.zeros(1)
        for i in range(currL**2):
            #print("meow2")
            #print(i)
            curr = currImgSplit[i]
            hist,bins = np.histogram(curr, bins=dict_size, density=True)
            #print(sum(hist))
            #print(hist.size)
            #hist_all = np.hstack((hist_all,hist*weights[l]))
            temp_hist = np.hstack((temp_hist,hist))
        
        #pdb.set_trace()

        temp_hist = np.delete(temp_hist,0)
        temp_hist = temp_hist/(currL**2)
        #print(sum(temp_hist))
        hist_all = np.hstack((hist_all,temp_hist*weights[l]))

        #np.delete(hist_all,0)

        #print("meow3")
        #print(weights[l])
    hist_all = np.delete(hist_all,0)
    #pdb.set_trace()
    #matplotlib.pyplot.bar(np.arange(len(hist_all)),hist_all)
    #matplotlib.pyplot.show()

    #finest_layer = [A for sub in np.array_split(
    #    wordmap,L, axis = 0) for A in np.array_split(sub,L, axis = 1)]
    #print(finest_layer)
    #pdb.set_trace()
    return hist_all

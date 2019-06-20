import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random
import tempfile

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    if len(image.shape) == 2:
        image = np.tile(image[:, newaxis], (1, 1, 3))

    if image.shape[2] == 4:
        image = image[:,:,0:3]

    image = skimage.color.rgb2lab(image)
    
    scales = [1,2,4,8,8*np.sqrt(2)]
    for i in range(len(scales)):
        for c in range(3):
            #img = skimage.transform.resize(image, (int(ss[0]/scales[i]),int(ss[1]/scales[i])),anti_aliasing=True)
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            if i == 0 and c == 0:
                imgs = img[:,:,np.newaxis]
            else:
                imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)

    return imgs

def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    filter_responses = extract_filter_responses(image)
    imgShape = np.shape(filter_responses)
    filter_responses_resize = filter_responses.reshape(
        imgShape[0]*imgShape[1],imgShape[2])
    diff = scipy.spatial.distance.cdist(filter_responses_resize, dictionary)
    #print(diff.shape)
    #print(diff)
    minRowVals = diff.min(axis=1)
    minRowIndex = np.argmin(diff,axis=1)    #diff[np.arange(len(diff)), np.argmin(diff, axis=1)]
    #print(minRowIndex)
    #print(minRowIndex.shape)

    wordmap = minRowIndex.reshape(imgShape[0],imgShape[1])
    #print(imgShape[0]*imgShape[1])

    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''


    i,alpha,image_path = args
    # ----- TODO -----
    image = skimage.io.imread(image_path)
    image = image.astype('float')/255
    filter_responses = extract_filter_responses(image)
    imgShape = np.shape(filter_responses)
    filter_responses_resize = filter_responses.reshape(
        imgShape[0]*imgShape[1],imgShape[2])
    filter_responses_random = np.random.permutation(filter_responses_resize)

    filter_output = filter_responses_random[0:alpha]
    #print('meow compdictone')
    #print(np.shape(filter_output))
    #outfile = tempfile.NamedTemporaryFile(prefix="zz")
    #np.savez(outfile)
    #print(outfile.name)
    #global namedTempFileList
    #namedTempFileList.append(outfile.name)
    #print(namedTempFileList)
    #print(outfile.name)
    currSaveFileName = "z" + str(i)
    np.save(currSaveFileName, filter_output)
    #proc_name = multiprocessing.current_process().proc_name
    #print('{0} image filter done by: {1}'.format(image_path,proc_name))

    pass

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''

    train_data = np.load("../data/train_data.npz")
    # ----- TODO -----
    filePathList = train_data['files']
    numImages = np.shape(filePathList)[0]
    #print(numImages)
    #print('meow compdict')

    alpha = 275
    K = 200
    #procs = []
    args_list = []
    #m = multiprocessing.Manager()
    #namedTempFileList = m.list()

    for i in range(numImages):
        currImgPath = "../data/" + filePathList[i]

        args = (i, alpha, currImgPath)
        #compute_dictionary_one_image(args)

        args_list.append(args)




        #proc = multiprocessing.Process(target=compute_dictionary_one_image,
        #    name=str(i), args=(i,alpha,currImgPath))

    
    #print(args_list)



    pool = multiprocessing.Pool(processes = num_workers)
    pool.map(compute_dictionary_one_image,args_list)


    totalResponses = np.empty((1, 60))
    #print(totalResponses.shape)

    for i in range(numImages):
        currData = np.load("z" + str(i) + ".npy")
        #print(currData.shape)
        totalResponses = np.concatenate((totalResponses, currData), axis=0)

    #print(totalResponses.shape)
    #print(totalResponses[0,:])
    totalResponses = np.delete(totalResponses,0,0)
    #print(totalResponses[0,:])
    #print(totalResponses.shape)
    #print(namedTempFileList)
    
    kmeans = sklearn.cluster.KMeans(n_clusters=K,n_jobs=num_workers).fit(totalResponses)
    dictionary = kmeans.cluster_centers_

    np.save("dictionary.npy", dictionary)

    pass

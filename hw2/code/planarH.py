import numpy as np
from numpy import matlib
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    # append ones to bottom row of p1, p2 so [x,y,1]'
    N = p1.shape[1]
    p1 = np.vstack((p1,np.ones(N,dtype=int)))
    p2 = np.vstack((p2,np.ones(N,dtype=int)))

    # A matrix
    A = np.zeros((2*N,9),dtype=int)
    #print(A)
    for i in range(N):
        x = p1[0,i]
        y = p1[1,i]
        u = p2[0,i]
        v = p2[1,i]
        #print(i)
        # odd - x rows
        A[(i*2)+1,:] = [ -u, -v, -1,  0, 0, 0,  x*u, x*v, x]
        #print(A)
        #even - y rows
        A[(i*2),:] = [ 0, 0, 0, -u, -v, -1, y*u,  y*v, y]
        #print(A)

    u,s,v = np.linalg.svd(A)
    #print(A)
    #w,v = np.linalg.eigh(np.matmul(np.transpose(A), A))
    #H2to1 = v[:,0].reshape(3,3)
    H2to1 = (v[-1,:]/v[-1,-1]).reshape(3,3)
    #H2to1 = np.transpose(v[:,0].reshape(3,3))
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    bestH = np.zeros((3,3),dtype=int)
    maxInliers = -1 
    numMatches = matches.shape[0]
    #print(numMatches)
    numPoints = 4
    # pull out corresponding locations from locs1 & locs2
    p1 = np.transpose(locs1[matches[:,0],0:2])
    p2 = np.transpose(locs2[matches[:,1],0:2])

    #print(p1.shape)
    #print(matches.shape[0])
    #print(p2.shape)
    numInliers = np.zeros(num_iter, dtype=int)
    allH = [None]*num_iter
    for i in range(num_iter):
        #randomly pick 4 points to calculate H
        randIndex = np.random.choice(numMatches,numPoints,replace=False)
        #print(randIndex)
        # get the respective random locations from p1 & p2
        x1 = p1[:,randIndex]
        x2 = p2[:,randIndex]
        #print(x1)
        #print(x2)

        # compute homography matrix
        currH = computeH(x1,x2)
        #print(p2.shape)
        p2_H = np.vstack((p2,np.ones((1,numMatches))))
        #print(p2_H.shape)

        p1_est = np.matmul( currH, p2_H )
        L = matlib.repmat(p1_est[2,:],2,1)
        p1_est = np.divide(p1_est[0:2,:],L)

        tempNumInliers = 0
        for j in range(numMatches):
            #currDist = np.linalg.norm(
            #    np.divide(p1_est[0:2,j],p1_est[2,j]) - p1[:,j])
            currDist = np.linalg.norm(p1_est[:,j] - p1[:,j])

            if currDist < tol:
                tempNumInliers +=1 

        numInliers[i] = tempNumInliers
        allH[i] = currH

    maxInliers = numInliers.max()
    bestH = allH[np.argmax(numInliers)]

    #print(maxInliers)
    #print(bestH)
    return bestH
        
    

if __name__ == '__main__':
    #im1 = cv2.imread('../data/model_chickenbroth.jpg')
    #im2 = cv2.imread('../data/chickenbroth_05.jpg')
    #im1 = cv2.imread('../data/pf_scan_scaled.jpg')
    #im2 = cv2.imread('../data/pf_floor.jpg')
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)


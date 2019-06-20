"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
from scipy import ndimage

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # normalize points
    pts1 = pts1/M
    pts2 = pts2/M
    n = pts1.shape[0] #num points

    x1 = pts1[:,0]
    x2 = pts2[:,0]
    y1 = pts1[:,1]
    y2 = pts2[:,1]

    A = np.vstack([
        x2*x1,
        x2*y1,
        x2,
        y2*x1,
        y2*y1,
        y2,
        x1,
        y1,
        np.ones(n)])

    V = np.linalg.svd(np.transpose(A))[2]
    F = V[-1,:].reshape(3,3)

    F = helper.refineF(F, pts1, pts2)
    F = helper._singularize(F)

    # un normalize data
    T = np.diag([1.0/M, 1.0/M, 1.0])
    F = np.dot(np.transpose(T), np.dot(F,T))

    # save results
    np.savez('../results/q2_1.npz', F=F, M=M)

    return F

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = pts1/M
    pts2 = pts2/M
    n = pts1.shape[0] #num points

    x1 = pts1[:,0]
    x2 = pts2[:,0]
    y1 = pts1[:,1]
    y2 = pts2[:,1]

    A = np.vstack([
        x2*x1,
        x2*y1,
        x2,
        y2*x1,
        y2*y1,
        y2,
        x1,
        y1,
        np.ones(n)])

    V = np.linalg.svd(np.transpose(A))[2]

    F1 = V[-1,:].reshape(3,3)
    F2 = V[-2,:].reshape(3,3)

    F1 = helper.refineF(F1, pts1, pts2)
    F2 = helper.refineF(F2, pts1, pts2)

    # given to find coefficients
    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)

    a0 = fun(0)
    a1 = 2*(fun(1) - fun(-1))/3 - (fun(2) - fun(-2))/12
    a2 = 0.5*fun(1) + 0.5*fun(-1) - fun(0) 
    #a3 = (fun(1) - fun(-1))/6 + (fun(2) - fun(-2))/12
    a3 = 0.5*(fun(1) - fun(-1)) - a1

    coeff = np.array([a3,a2,a1,a0])
    alpha = np.polynomial.polynomial.polyroots(coeff)
    # extract real solutions
    alpha = np.real(alpha[np.isreal(alpha)])

    T = np.diag([1.0/M, 1.0/M, 1.0])

    Farray = []
    for a in alpha:
        #a = a.real
        F = a*F1 +(1-a)*F2
        F = helper._singularize(F)
        F = helper.refineF(F, pts1, pts2)
        F = np.dot(np.transpose(T), np.dot(F,T))

        Farray.append(F)

    Farray = np.array(Farray)

    np.savez('../results/q2_2.npz', F=F, M=M, pts1=pts1, pts2=pts2)
    return Farray

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    return np.dot(np.transpose(K2), np.dot(F, K1))


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    n = pts1.shape[0]
    x1 = pts1[:,0]
    x2 = pts2[:,0]
    y1 = pts1[:,1]
    y2 = pts2[:,1]

    # initialize values
    P = np.zeros((n,3))
    err = 0

    for i in range(n):
        A = np.array([x1[i]*C1[2,:] - C1[0,:],
                      y1[i]*C1[2,:] - C1[1,:],
                      x2[i]*C2[2,:] - C2[0,:],
                      y2[i]*C2[2,:] - C2[1,:]])

        V = np.linalg.svd(A)[2]
        p = V[-1,:]
        p = np.divide(p,p[3])
        P[i,:] = p[0:3]

        proj1 = np.dot(C1, p)
        proj2 = np.dot(C2, p)
        err1 = np.linalg.norm(proj1[:2]/proj1[-1] - pts1[i])**2
        err2 = np.linalg.norm(proj2[:2]/proj2[-1] - pts2[i])**2
        err += err1 + err2

    #print(err)
    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation


    point = np.array([x1, y1, 1])
    epiLine = np.dot(F,point)
    epiLine = np.divide(epiLine, np.linalg.norm(epiLine))
    a, b, c = epiLine[0:3]

    h,w = im1.shape[0:2]

    step = 25 # steps around to search
    winSize = 10 # window size
    min_dist = np.inf

    # ax + by + c = 0
    ly = np.arange(y1-step, y1+step)
    lx = (-b*ly - c)/a

    # get patch around point in im1
    patch1 = im1[y1-winSize : y1+winSize+1, x1-winSize : x1+winSize+1, :]

    for i in range(len(lx)):
        #get patch around point in im2
        currX2 = np.int(lx[i])
        currY2 = ly[i]
        patch2 = im2[currY2-winSize : currY2+winSize+1, 
            currX2-winSize : currX2+winSize+1, :]

        if patch1.shape != patch2.shape:
            continue

        # calculate euclidiean distance
        dist = np.linalg.norm(patch1 - patch2)

        # apply gaussian filter
        dist = ndimage.gaussian_filter(dist, sigma=1, output=np.float64)

        if dist < min_dist:
            min_dist = dist
            x2 = currX2
            y2 = currY2

    return x2, y2

import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    pano_im = cv2.warpPerspective(
        im2.astype(np.float32), H2to1.astype(np.float32).reshape(3,3), (
            im2.shape[0]*3, im2.shape[1]))

    cv2.imshow('warped image', pano_im)
    cv2.imwrite('../results/6_1.jpg', pano_im)
    cv2.destroyAllWindows()
    
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    im1H = im1.shape[0]
    im1W = im1.shape[1]
    im2H = im2.shape[0]
    im2W = im2.shape[1]

    im1H_max = im1H -1
    im1W_max = im1W -1
    im2H_max = im2H -1
    im2W_max = im2W -1

    corner = np.array(
        [[0,0,im2W_max,im2W_max], [0,im2H_max, im2H_max,0], [1,1,1,1]])

    warpCorner = np.matmul(H2to1.reshape(3,3),corner)
    #print(warpCorner.shape)
    warpCorner_norm = np.divide(warpCorner[0:2,:], warpCorner[2,:])

    out_size = (int(
        max(im1W_max, np.max(warpCorner_norm[0])) - min(
            0, np.min(warpCorner_norm[0]))), int(
        max(im1H_max, np.max(warpCorner_norm[1])) - min(
            0, np.min(warpCorner_norm[1]))))

    M = np.array([[1,0,int(abs(min(0, np.min(warpCorner_norm[0]))))],
        [0,1, int(abs(min(0, np.min(warpCorner_norm[1]))))],
        [0,0,1]],dtype=np.float32)

    warp_im1 = cv2.warpPerspective(np.asarray(im1,dtype=np.float32), M, out_size)
    warp_im2 = cv2.warpPerspective(
        np.asarray(im2,dtype=np.float32), np.asarray(
            np.matmul(M,H2to1.reshape(3,3)),dtype=np.float32), out_size )

    pano_im  = np.maximum(warp_im1, warp_im2)

    return pano_im

def generatePanorama(im1,im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    im3 = imageStitching_noClip(im1, im2, H2to1)

    return im3

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    #print(im1.shape)
    #locs1, desc1 = briefLite(im1)
    #locs2, desc2 = briefLite(im2)
    #matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    #H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    Hfile = '../results/H2to1.npy'
    #np.save= (Hfile,[H2to1])
    #np.save('../results/H2to1.npy', [H2to1])
    #H2to1 = np.load('../results/H2to1.npy')
    H2to1 = np.load(Hfile)
    #imageStitching(im1,im2,H2to1)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    print(H2to1)
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    im3 = generatePanorama(im1,im2)
    cv2.imshow('q6_3', im3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
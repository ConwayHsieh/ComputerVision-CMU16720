import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def LucasKanadeAffine(It, It1):
    # Input: 
    #   It: template image
    #   It1: Current image
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)

    w = It.shape[1]
    h = It.shape[0]

    xx, yy = np.meshgrid(np.arange(h),np.arange(w))
    xx = xx.reshape(1,-1)
    yy = yy.reshape(1,-1)

    It_dy, It_dx = np.gradient(It)

    #create mask for warping
    mask = np.ones((h,w))

    thresh = 0.01

    #initialize dp so that its norm is above 0
    dp = np.array((0.1,0.1),dtype=np.float64)

    while np.linalg.norm(dp) >= thresh:
        warp_img = cv2.warpAffine(It, M, (w,h))
        warp_dx = cv2.warpAffine(It_dx, M, (w,h)).reshape(1,-1)
        warp_dy = cv2.warpAffine(It_dy, M, (w,h)).reshape(1,-1)
        warp_mask = cv2.warpAffine(mask, M, (w,h))

        #print(warp_mask)

        #warp It1 into It
        mask_img = warp_mask * It1  

        #print((warp_dx*xx).shape)
        #A & b
        A = np.hstack(((warp_dx*xx).reshape(-1,1), 
            (warp_dx*yy).reshape(-1,1), 
            warp_dx.reshape(-1,1),
            (warp_dy*xx).reshape(-1,1), 
            (warp_dy*yy).reshape(-1,1), 
            (warp_dy).reshape(-1,1)))
        #A = warp_dx*xx
        b = warp_img - mask_img
        b = b.reshape(-1,1)

        #print(A.shape)
        #print(b.shape)

        dp = np.linalg.lstsq(A,b, rcond=None)[0].reshape(6)

        p += dp

        M[0,:] = [1.0 + p[0], p[1], p[2]]
        M[1,:] = [p[3], 1.0 + p[4], p[5]]

    return M

import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    #p = np.zeros(6)

    w = It.shape[1]
    h = It.shape[0]

    xx, yy = np.meshgrid(np.arange(h),np.arange(w))
    xx = xx.reshape(1,-1)
    yy = yy.reshape(1,-1)

    It_dy, It_dx = np.gradient(It)
    It_dx = It_dx.flatten()
    It_dy = It_dy.flatten()

    #create mask for warping
    mask = np.ones((h,w))


    A = np.hstack(((It_dx*xx).reshape(-1,1), 
            (It_dx*yy).reshape(-1,1), 
            It_dx.reshape(-1,1),
            (It_dy*xx).reshape(-1,1), 
            (It_dy*yy).reshape(-1,1), 
            (It_dy).reshape(-1,1)))

    A_inv = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A))

    thresh = 0.01

    #initialize dp so that its norm is above 0
    dp = np.array((0.1,0.1),dtype=np.float64)

    while np.linalg.norm(dp) >= thresh:

        warp_img = cv2.warpAffine(It, M, (w,h))
        warp_mask = cv2.warpAffine(mask, M, (w,h))

        #print(warp_mask)
        #print((warp_dx*xx).shape)
        #A & b
        
        #warp It1 into It
        mask_img = warp_mask * It1  
        
        #A = warp_dx*xx
        #b = warp_img - mask_img
        b = mask_img - warp_img
        b = b.reshape(-1,1)

        #print(A.shape)
        #print(b.shape)

        #dp = np.linalg.lstsq(A,b, rcond=None)[0].reshape(6)
        dp = np.dot(A_inv, b)
        #p += dp

        #dM = np.array([[1.0 + dp[0], dp[1], dp[2]], 
        #    [dp[3], 1.0 + dp[4], dp[5]], 
        #    [0.0, 0.0, 1.0]], dtype=np.float64)

        dM = np.vstack((dp.reshape((2,3)) + np.eye(3,3)[0:2,:], np.eye(3,3)[2,:]))

        dM = np.linalg.inv(dM)

        M = np.dot(M, dM)

    return M

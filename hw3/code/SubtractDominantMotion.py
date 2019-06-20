import numpy as np
import cv2
from scipy.ndimage.morphology import binary_dilation, binary_erosion

from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    mask = np.zeros(image1.shape, dtype=bool)

    # compute M for It & It+1
    #M = LucasKanadeAffine(image1, image2)
    M = InverseCompositionAffine(image1, image2)

    w = image1.shape[1]
    h = image1.shape[0]

    # Warp It using M in order to register to It+1
    warp_img1 = cv2.warpAffine(image1, M, (w,h))

    # Subtract It from It+1
    diff = np.abs(image2 - warp_img1)

    thresh = 0.3

    # find where difference exceeds threshold
    mask[diff > thresh] = 1

    #dilate for better performance
    mask = binary_dilation(mask,structure=np.ones((5,5)))

    return mask

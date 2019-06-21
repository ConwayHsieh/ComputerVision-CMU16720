import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    
    # denoise
    #image = skimage.restoration.denoise_bilateral(image, multichannel = True)
    gblur = skimage.filters.gaussian(image, sigma=2, multichannel=True)
    
    #greyscale
    grey = skimage.color.rgb2grey(gblur)
    
    #threshold
    thresh = skimage.filters.threshold_otsu(grey)
    
    #morphology
    bw = skimage.morphology.closing( grey < thresh, 
    	skimage.morphology.square(10))

    #remove artifacts at image border
    cleared = skimage.segmentation.clear_border(bw)

    #label
    label_image = skimage.measure.label(cleared)

    #skip small boxes
    for region in skimage.measure.regionprops(label_image):
    	if region.area >= 200:
    		# save large regions
    		currBox = region.bbox
    		bboxes.append(currBox)

    return bboxes, bw
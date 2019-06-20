import numpy as np
import cv2

np.set_printoptions(threshold=np.inf)


def createGaussianPyramid(im, sigma0=1,
                          k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4]):
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max() > 10:
        im = np.float32(im) / 255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0 * k**i
        im_pyramid.append(cv2.GaussianBlur(im, (0, 0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(
        im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1, 0, 1, 2, 3, 4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    #DoG_pyramid = []
    DoG_levels = levels[1:]
    ################
    # TO DO ...
    # compute DoG_pyramid here
    h, w, l = gaussian_pyramid.shape
    DoG_pyramid = np.zeros((h, w, l - 1))
    #print(DoG_pyramid.shape)
    for i in range(l - 1):
        DoG_pyramid[:, :, i] = gaussian_pyramid[
            :, :, i + 1] - gaussian_pyramid[:, :, i]

    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid

    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    # TO DO ...
    # Compute principal curvature here
    h, w, l = DoG_pyramid.shape
    principal_curvature = np.zeros((h, w, l))
    for i in range(l):
        currLayer = DoG_pyramid[:, :, i]
        Dxx = cv2.Sobel(currLayer, cv2.CV_64F, dx=2, dy=0, ksize=3)
        Dyy = cv2.Sobel(currLayer, cv2.CV_64F, dx=0, dy=2, ksize=3)
        Dxy = cv2.Sobel(currLayer, cv2.CV_64F, dx=1, dy=1, ksize=3)

        tr = Dxx + Dyy
        det = np.multiply(Dxx, Dyy) - np.square(Dxy)
        np.seterr(divide='ignore', invalid='ignore')
        principal_curvature[:, :, i] = np.divide(np.square(tr), det)*2  

    return principal_curvature




def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
                    th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = np.zeros((1,3),dtype=int)
    #print(locsDoG)
    ##############
    #  TO DO ...
    # Compute locsDoG here
    #temp =0
    H, W, L = DoG_pyramid.shape
    for h in range(H):
        if h == 0 or h == H-1:
            continue
        
        for w in range(W):
            if w == 0 or w == W-1:
                continue
            
            for l in range(L):
                # check if curvature too large
                currCurv = principal_curvature[h, w, l]
                if currCurv >= th_r:
                    continue

                # check if contrast too large
                currDoGPoint = DoG_pyramid[h, w, l]
                if np.abs(currDoGPoint) <= th_contrast:
                    continue 

                #temp +=1
                currNeighbors = DoG_pyramid[h-1:h+2, w-1:w+2, l]
                #print(currNeighbors.shape)
                #print(currNeighbors)
                if l == 0:  # no scale above
                    currNeighbors[1, 1] = DoG_pyramid[h, w, l + 1]
                    currNeighbors = currNeighbors.reshape(1, 9)
                elif l == L-1:  # no scale below
                    currNeighbors[1, 1] = DoG_pyramid[h, w, l - 1]
                    currNeighbors = currNeighbors.reshape(1, 9)
                else:  # all 10 neighbors
                    currNeighbors[1, 1] = DoG_pyramid[h, w, l - 1]
                    #print(currNeighbors)
                    currNeighbors = np.append(
                        currNeighbors.reshape(1, 9),np.array(DoG_pyramid[h, w, l + 1]))
                    #print(currNeighbors)

                #print(currNeighbors)
                # check if extrema
                if (currDoGPoint > np.amax(currNeighbors)) or (
                    currDoGPoint < np.amin(currNeighbors)):
                    locsDoG = np.vstack((locsDoG,np.array([w, h, l])))

    locsDoG = np.delete(locsDoG, 0,0)
    #print(temp)
    #print(locsDoG.shape)
    return locsDoG


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4],
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid,levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1, 0, 1, 2, 3, 4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #print(DoG_pyr)
    #print(DoG_levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    #print(pc_curvature)
    #print(pc_curvature.shape)
    displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    print(locsDoG.shape)

    cv2.imshow('image',im)
    r,c = locsDoG.shape
    for i in range(r):
        cv2.circle(im,(int(locsDoG[i,0]),int(locsDoG[i,1])),1,(0,255,0),-1)
    cv2.imshow('image',im)
    cv2.imwrite('../results/q1.5.png',im)
    cv2.waitKey(0)  # press any key to exit
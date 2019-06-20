import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here

    p = np.zeros([2,1])
    B = bases.reshape(-1,bases.shape[2])
    B0 = np.eye(B.shape[0]) - np.dot(B,np.transpose(B))

	# pull out corners of rectangle
    x1,y1,x2,y2 = rect.astype(int)

    # spline the template and current images
    interp_spline_It1 = RectBivariateSpline(
        np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    interp_spline_It = RectBivariateSpline(
        np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    # set up meshgrid to 
    xx, yy = np.meshgrid(np.arange(x1, x2+1), np.arange(y1,y2+1))
    
    # need to flatten meshgrid as input
    xx = xx.flatten()
    yy = yy.flatten()

    # need initial interpolation for comparison
    I_init = interp_spline_It.ev(yy,xx, dx=0, dy=0)

    #initialize dp so that its norm is above 0
    dp = np.array((0.1,0.1),dtype=np.float64)

    # set threshold for delta p norm
    thresh = 0.01

    # loop until threshold reached
    while np.linalg.norm(dp) >= thresh:
        # axes are flipped for interpolation
        x = xx + p[1]
        y = yy + p[0]

        I  = interp_spline_It1.ev(y, x, dx=0, dy=0)
        Ix = interp_spline_It1.ev(y, x, dx=1, dy=0).reshape(-1,1)
        Iy = interp_spline_It1.ev(y, x, dx=0, dy=1).reshape(-1,1)

        A = np.hstack((Ix, Iy))
        b = I_init - I

        #apply bases matrix
        A0 = np.dot(B0,A)
        b0 = np.dot(B0,b.reshape(-1,1))

        dp = np.linalg.lstsq(A0,b0,rcond=None)[0]
        p += dp

    return p
    

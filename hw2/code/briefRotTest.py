import cv2
import numpy as np
import matplotlib.pyplot as plt
from BRIEF import briefLite, briefMatch


im = cv2.imread('../data/model_chickenbroth.jpg')

inc = 10
#cv2.getRotationMatrix2D()
#cv2.warpAffine()

h,w = im.shape[:2]
centerX, centerY = (h//2, w//2)
center = (centerX,centerY)

angle = np.arange(0,360,10,dtype=int)
numAngles = len(angle)
numMatches = np.zeros(numAngles, dtype=int)
locs1, desc1 = briefLite(im)
#locs2, desc2 = briefLite(im2)
#matches = briefMatch(desc1, desc2)

for i in range(numAngles):
	currAngle = angle[i]

	a = cv2.getRotationMatrix2D(center,currAngle,1.0)
	rotIm = cv2.warpAffine(im, a, (h,w))
	locs2, desc2 = briefLite(rotIm)
	matches = briefMatch(desc1,desc2)

	numMatches[i] = len(matches)



plt.bar(angle, numMatches, align='center', alpha=0.5)
plt.show()
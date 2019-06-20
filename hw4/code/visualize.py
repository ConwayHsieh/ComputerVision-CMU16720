'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import matplotlib.pyplot as plt
import submission, helper, findM2
from mpl_toolkits.mplot3d import Axes3D

# load data
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
data = np.load('../data/some_corresp.npz')
intr = np.load('../data/intrinsics.npz')
coord = np.load('../data/templeCoords.npz')

# extract variables
pts1 = data['pts1']
pts2 = data['pts2']
K1 = intr['K1']
K2 = intr['K2']
x1 = coord['x1'][:,0]
y1 = coord['y1'][:,0]

M = 640
n = x1.shape[0]


F = submission.eightpoint(pts1, pts2, M)

pts1_new = np.transpose(np.vstack((x1, y1)))
pts2_new = np.zeros(pts1_new.shape)
for i in range(n):
	x2, y2 = submission.epipolarCorrespondence(im1, im2, F, 
		x1[i], y1[i])
	pts2_new[i,:] = np.array([x2, y2])
#print(pts1_new.shape)
#print(pts2_new.shape)

# calculate essential matrix
E = submission.essentialMatrix(F, K1, K2)

M2s = helper.camera2(E)

M1 =  np.hstack(( np.eye(3, dtype=int), np.zeros((3,1), dtype=int) ))
C1 = np.dot(K1, M1)

M2 = None
C2 = None
P = None
minErr = np.inf

for i in range(M2s.shape[2]):
	#print(i)
	currM2 = M2s[:,:,i]
	#print(currM2)
	currC2 = np.dot(K2, currM2)
	currP, currErr = submission.triangulate(C1, pts1_new, currC2, pts2_new)
	#print(currErr)

	# make sure Z values are positive for this M2
	if not np.all(currP[:,2] >= 0): 
		continue

	if currErr < minErr:
		minErr = currErr
		M2 = currM2
		C2 = currC2
		P = currP

#print(M2)
np.savez('../results/q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)

f = plt.figure()
ax = Axes3D(f)
ax.scatter(P[:,0], P[:,1], P[:,2], color='b', marker='.')
plt.show()

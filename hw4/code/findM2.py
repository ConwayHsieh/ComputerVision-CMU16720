'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import submission, helper

# load data
data = np.load('../data/some_corresp.npz')
intr = np.load('../data/intrinsics.npz')

# extract variables
pts1 = data['pts1']
pts2 = data['pts2']
K1 = intr['K1']
K2 = intr['K2']

M = 640

# calculate Fundamental matrix
F = submission.eightpoint(pts1, pts2, M)

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
	currP, currErr = submission.triangulate(C1, pts1, currC2, pts2)
	#print(currErr)

	if len(currP[currP[:, 2] > 0]) != currP.shape[0]: 
		continue

	if currErr < minErr:
		minErr = currErr
		M2 = currM2
		C2 = currC2
		P = currP

#print(minErr)
np.savez('../results/q3_3.npz', M2=M2, C2=C2, P=P)
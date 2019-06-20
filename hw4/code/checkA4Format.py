"""
Check the dimensions of function arguments
This is *not* a correctness check

Written by Chen Kong, 2018.
"""
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

intr = np.load('../data/intrinsics.npz')
K1 = intr['K1']
K2 = intr['K2']

N = data['pts1'].shape[0]
M = 640

# 2.1
F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'
print('F8')
print(F8)
#helper.displayEpipolarF(im1, im2, F8)


# 2.2
#F7 = sub.sevenpoint(data['pts1'][:7, :], data['pts2'][:7, :], M)
F7 = sub.sevenpoint(data['pts1'][70:77, :], data['pts2'][70:77, :], M)
assert (len(F7) == 1) | (len(F7) == 3), 'sevenpoint returns length-1/3 list'

for f7 in F7:
	assert f7.shape == (3, 3), 'seven returns list of 3x3 matrix'

#print(F7.shape)
print('F7')
print(F7)
#helper.displayEpipolarF(im1, im2, F7[0])

# 3.1
print('E')
print(sub.essentialMatrix(F8, K1, K2))

C1 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
C2 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)

P, err = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
print(err)
assert P.shape == (N, 3), 'triangulate returns Nx3 matrix P'
assert np.isscalar(err), 'triangulate returns scalar err'

# 4.1
x2, y2 = sub.epipolarCorrespondence(im1, im2, F8, data['pts1'][0, 0], data['pts1'][0, 1])
assert np.isscalar(x2) & np.isscalar(y2), 'epipolarCorrespondence returns x & y coordinates'
np.savez('../results/q4_1.npz', F=F8, pts1=data['pts1'], pts2=data['pts2'])
helper.epipolarMatchGUI(im1,im2,F8)


print('Format check passed.')

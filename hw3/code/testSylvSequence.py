import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanade import LucasKanade
from LucasKanadeBasis import LucasKanadeBasis

rect = np.array([101,61,155,107],dtype=np.float64).reshape(4,1)
rect1 = rect.copy()

frames = np.load('../data/sylvseq.npy')
numFrames = frames.shape[2]
saveFrames = [1,200,300,350,400]
bases = np.load('../data/sylvbases.npy')

w = rect[2] - rect[0]
h = rect[3] - rect[1]

#stores all values of rect during tracking
rectMat = rect

# start figure
fig,ax = plt.subplots(1)
ax.set_title("Lucas Kanade Tracking Animation w/ Appearance Bias")

for i in range(numFrames-1):
	#skip i = 0
	#if i == 0:
	#	continue
	# extract 2 frames
	currFrame = frames[:,:,i]
	nextFrame = frames[:,:,i+1]

	# caluclate p
	p = LucasKanadeBasis(currFrame, nextFrame, rect, bases)
	p1 = LucasKanade(currFrame, nextFrame, rect1)
	#if p[1] > 1:
		#break
	#print(p)
	#print("meow")
	rect += np.array([p[1], p[0], p[1], p[0]])
	rect1 += np.array([p1[1], p1[0], p1[1], p1[0]]).reshape(4,1)
	#print(rect)
	rectMat = np.vstack((rectMat,rect))

	currPatch1 = patches.Rectangle((rect1[0],rect1[1]),w,h,linewidth=1,
		edgecolor='red',fill=False)
	ax.add_patch(currPatch1)

	currPatch = patches.Rectangle((rect[0],rect[1]), w, h, linewidth=1, 
		edgecolor='blue', fill=False)
	ax.add_patch(currPatch)

	plt.imshow(nextFrame,cmap='gray')

	if i in saveFrames:
		plt.savefig("LK_AppearanceBasis_"+str(i)+".png")

	plt.pause(0.01)
	ax.clear()

np.save('sylvseqrects.npy', rectMat)
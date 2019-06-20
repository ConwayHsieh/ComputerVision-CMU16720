import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
#import Lucas Kanade function
from LucasKanade import LucasKanade

frames = np.load('../data/carseq.npy')
numFrames = frames.shape[2]
saveFrames = [1,100,200,300,400]

#given rect coordinates
rect = np.array([59,116,145,151],dtype=np.float64)

# calculate width and height of rectangle for plotting tracking rectangle
w = rect[2] - rect[0]
h = rect[3] - rect[1]

#stores all values of rect during tracking
rectMat = rect

# start figure
fig,ax = plt.subplots(1)
ax.set_title("Lucas Kanade Tracking Animation")

for i in range(numFrames-1):
	#skip i = 0
	#if i == 0:
	#	continue
	# extract 2 frames
	currFrame = frames[:,:,i]
	nextFrame = frames[:,:,i+1]

	# caluclate p
	p = LucasKanade(currFrame, nextFrame, rect)
	#if p[1] > 1:
		#break
	#print(p)
	#print("meow")
	rect += np.array([p[1], p[0], p[1], p[0]])
	#print(rect)
	rectMat = np.vstack((rectMat,rect))

	currPatch = patches.Rectangle((rect[0],rect[1]), w, h, linewidth=1, 
		edgecolor='red', fill=False)
	ax.add_patch(currPatch)

	plt.imshow(nextFrame,cmap='gray')

	if i in saveFrames:
		plt.savefig("LK_"+str(i)+".png")

	plt.pause(0.01)
	ax.clear()

np.save('carseqrects.npy', rectMat)

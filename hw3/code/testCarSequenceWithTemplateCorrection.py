import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanade import LucasKanade

frames = np.load('../data/carseq.npy')
frame0 = frames[:,:,0]
numFrames = frames.shape[2]
saveFrames = [1,100,200,300,400]
sigma = 5

LK_rect_All = np.load('carseqrects.npy')

#given rect coordinates
orig_rect = np.array([59,116,145,151],dtype=np.float64)
rect = orig_rect.copy()


# calculate width and height of rectangle for plotting tracking rectangle
w = rect[2] - rect[0]
h = rect[3] - rect[1]

#stores all values of rect during tracking
rectMat = rect

# start figure
fig,ax = plt.subplots(1)
ax.set_title("Lucas Kanade Tracking Animation with Template Correction")

for i in range(numFrames-1):
	#skip i = 0
	#if i == 0:
	#	continue
	# extract 2 frames
	currFrame = frames[:,:,i]
	nextFrame = frames[:,:,i+1]

	# caluclate p
	p = LucasKanade(currFrame, nextFrame, rect)

	p0 = np.array([rect[1]+p[0] - orig_rect[1], rect[0]+p[1]-orig_rect[0]])

	pstar = LucasKanade(frame0, nextFrame, orig_rect, p0)

	if np.linalg.norm(pstar-p0) < sigma:
		rect += np.array([p[1], p[0], p[1], p[0]])
	else:
		rect = np.array([orig_rect[0]+ pstar[1], orig_rect[1]+pstar[0],
			rect[0]+w, rect[1]+h])

	rectMat = np.vstack((rectMat,rect))

	currPatch = patches.Rectangle((rect[0],rect[1]), w, h, linewidth=1, 
		edgecolor='blue', fill=False)
	ax.add_patch(currPatch)

	LK_rect = LK_rect_All[i,:]
	LK_w = LK_rect[2]-LK_rect[0]
	LK_h = LK_rect[3]-LK_rect[1]

	currLKPatch = patches.Rectangle((LK_rect[0],LK_rect[1]),LK_w, LK_h,
		linewidth=1, edgecolor='red',fill=False)
	ax.add_patch(currLKPatch)

	plt.imshow(nextFrame,cmap='gray')

	if i in saveFrames:
		plt.savefig("LK_TemplateCorrection_"+str(i)+".png")

	plt.pause(0.01)
	ax.clear()

np.save('carseqrects-wcrt.npy', rectMat)
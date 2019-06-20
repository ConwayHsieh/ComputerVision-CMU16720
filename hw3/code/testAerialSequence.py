import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from SubtractDominantMotion import SubtractDominantMotion
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

frames = np.load('../data/aerialseq.npy')
numFrames = frames.shape[2]
saveFrames = [30, 60, 90, 120]

fig,ax = plt.subplots(1)
ax.set_title("Moving Object Detection")

for i in range(numFrames-1):
    print('frame:', i)
    # extract 2 frames
    currFrame = frames[:,:,i]
    nextFrame = frames[:,:,i+1]

    mask = SubtractDominantMotion(currFrame, nextFrame)

    img = np.copy(nextFrame)

    img = np.stack((img, img, img), axis=2)
    img[:,:,2][mask==1] = 1

    plt.imshow(img)



    #if i in saveFrames:
        #plt.savefig("MovingObjectDetection_"+str(i)+".png")

    plt.pause(0.01)
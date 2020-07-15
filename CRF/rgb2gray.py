import sys
import cv2
import numpy as np

oriname=sys.argv[1]
oriimg=cv2.imread(oriname)
#oriimg=cv2.resize(oriimg,(320,240))
#print oriimg.shape
orimat=np.asarray(oriimg)
#print(type(oriimg))

matgray=np.zeros([480,640], dtype=np.uint8)
#print matrgb.shape

mask=orimat==(139,0,0)
#print mask
mask=mask[:,:,0:1]
#print mask.shape
mask=np.squeeze(mask)
#print mask.shape
matgray[mask]=1

mask=orimat==(0,0,128)
mask=mask[:,:,0:1]
mask=np.squeeze(mask)
#print mask
matgray[mask]=57

cv2.imwrite(sys.argv[2],matgray)

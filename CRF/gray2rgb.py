import sys
import cv2
import numpy as np

oriname=sys.argv[1]
oriimg=cv2.imread(oriname,0)
#oriimg=cv2.resize(oriimg,(320,240))
#print oriimg.shape
orimat=np.asarray(oriimg)
#print(type(oriimg))

matrgb=np.ones([480,640,3], dtype=np.uint8)
#print matrgb.shape

matrgb[:,:,:]=255
#print matrgb

matrgb[orimat==1]=(139,0,0)
#print matrgb
matrgb[orimat==57]=(0,0,128)
#print matrgb

#matrgb=np.int8(matrgb)
#rgbimg=cv2.fromarray(array)
cv2.imwrite(sys.argv[2],matrgb)

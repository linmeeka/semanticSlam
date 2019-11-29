import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib

print 'Initializing Mask RCNN network...'
# Root directory of the project
ROOT_DIR = os.getcwd()
ROOT_DIR = "./src/python"
print(ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Set batch size to 1 since we'll be running inference on
# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
       'bus', 'train', 'truck', 'boat', 'traffic light',
       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
       'kite', 'baseball bat', 'baseball glove', 'skateboard',
       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
       'teddy bear', 'hair drier', 'toothbrush']
class_colors = {'BG':0., 'person':0., 'bicycle':0., 'car':0., 'motorcycle':0., 'airplane':0.,
       'bus':0., 'train':0., 'truck':0., 'boat':0., 'traffic light':0.,
       'fire hydrant':0., 'stop sign':0., 'parking meter':0., 'bench':0., 'bird':0.,
       'cat':0., 'dog':0., 'horse':0., 'sheep':0., 'cow':0., 'elephant':0., 'bear':0.,
       'zebra':0., 'giraffe':0., 'backpack':0., 'umbrella':0., 'handbag':0., 'tie':0.,
       'suitcase':0., 'frisbee':0., 'skis':0., 'snowboard':0., 'sports ball':0.,
       'kite':0., 'baseball bat':0., 'baseball glove':0., 'skateboard':0.,
       'surfboard':0., 'tennis racket':0., 'bottle':0., 'wine glass':0., 'cup':0.,
       'fork':0., 'knife':0., 'spoon':0., 'bowl':0., 'banana':0., 'apple':0.,
       'sandwich':0., 'orange':0., 'broccoli':0., 'carrot':0., 'hot dog':0., 'pizza':0.,
       'donut':0., 'cake':0., 'chair':0., 'couch':0., 'potted plant':0., 'bed':0.,
       'dining table':0., 'toilet':0., 'tv':0., 'laptop':0., 'mouse':0., 'remote':0.,
       'keyboard':0., 'cell phone':0., 'microwave':0., 'oven':0., 'toaster':0.,
       'sink':0., 'refrigerator':0., 'book':0., 'clock':0., 'vase':0., 'scissors':0.,
       'teddy bear':0., 'hair drier':0., 'toothbrush':0.}
print 'Initialated Mask RCNN network...'

def GetDynSeg(image,image2=None):
	h = image.shape[0]
	w = image.shape[1]
	if len(image.shape) == 2:
		im = np.zeros((h,w,3))
		im[:,:,0]=image
		im[:,:,1]=image
		im[:,:,2]=image
		image = im
	#if image2 is not None:
	#	args+=[image2]
	# Run detection
	results = model.detect([image], verbose=0)
	# Visualize results
	r = results[0]
	i = 0
	mask = np.zeros((h,w))
	for roi in r['rois']:
    	image_m = r['masks'][:,:,i]
		mask[image_m == 1] = class_colors[class_names[r['class_ids'][i]]]
		print class_colors[class_names[r['class_ids'][i]]]
		'''
		if class_names[r['class_ids'][i]] == 'person':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'bicycle':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'car':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'motorcycle':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'airplane':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'bus':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'train':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'truck':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'boat':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'bird':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'cat':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'dog':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'horse':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'sheep':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'cow':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'elephant':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'bear':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'zebra':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if class_names[r['class_ids'][i]] == 'giraffe':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.		
		'''
		i+=1
	#print('GetSeg mask shape:',mask.shape)

	return mask

im = np.zeros((480,640,3))
mask = GetDynSeg(im)
print("Mask R-CNN is correctly working")


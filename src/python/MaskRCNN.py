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
# import visualize

current_segmentation = None
current_class_ids = None
current_bounding_boxes = None
current_score=None

class Mask:
    """
    """
    def __init__(self):
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
	self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO
	self.model.load_weights(COCO_MODEL_PATH, by_name=True)

	# pre-paremeters
	self.SCORE_T = 0
	self.FILTER_CLASSES = ['person']
	self.FILTER_WEIGHTS = {'person': 1} #{'person': 255}
	self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
	self.class_colors = {'BG':0., 'person':1., 'bicycle':2., 'car':3., 'motorcycle':4., 'airplane':5.,
       'bus':6., 'train':7., 'truck':8., 'boat':9., 'traffic light':10.,
       'fire hydrant':11., 'stop sign':12., 'parking meter':13., 'bench':14., 'bird':15.,
       'cat':15., 'dog':17., 'horse':18., 'sheep':19., 'cow':20., 'elephant':21., 'bear':22.,
       'zebra':23., 'giraffe':24., 'backpack':25., 'umbrella':26., 'handbag':27., 'tie':28.,
       'suitcase':29., 'frisbee':30., 'skis':31., 'snowboard':32., 'sports ball':33.,
       'kite':34., 'baseball bat':35., 'baseball glove':36., 'skateboard':37.,
       'surfboard':38., 'tennis racket':39., 'bottle':40., 'wine glass':41., 'cup':42.,
       'fork':43., 'knife':44., 'spoon':45., 'bowl':46., 'banana':47., 'apple':48.,
       'sandwich':49., 'orange':50., 'broccoli':51., 'carrot':52., 'hot dog':53., 'pizza':54.,
       'donut':55., 'cake':56., 'chair':57., 'couch':58., 'potted plant':59., 'bed':60.,
       'dining table':61., 'toilet':62., 'tv':63., 'laptop':64., 'mouse':65., 'remote':66.,
       'keyboard':67., 'cell phone':68., 'microwave':69., 'oven':70., 'toaster':71.,
       'sink':72., 'refrigerator':73., 'book':74., 'clock':75., 'vase':76., 'scissors':77.,
       'teddy bear':78., 'hair drier':79., 'toothbrush':80.}

	self.FILTER_CLASSES = [self.class_names.index(x) for x in self.FILTER_CLASSES]
	self.FILTER_WEIGHTS = {self.class_names.index(x): self.FILTER_WEIGHTS[x] for x in self.FILTER_WEIGHTS}
    print 'Initialated Mask RCNN network...'
    
    def GetSegResult(self,image,image2=None):
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
	results = self.model.detect([image], verbose=0)
	# Visualize results
	r = results[0]
	masks = r['masks']
        scores = r['scores']
   	class_ids = r['class_ids']
	rois = r['rois']
	n = len(class_ids)
	
	id_image = np.zeros([h,w], np.uint8)
        exported_class_ids = []
        exported_rois = []
	exported_scores= []
	print "class_ids:"
	print class_ids
	for m in range(n):
            class_id = class_ids[m]
	    if True:
            #if len(self.FILTER_CLASSES) == 0 or class_id in self.FILTER_CLASSES:
                if scores[m] >= self.SCORE_T:
                    mask = masks[:,:,m]
                    val = len(exported_class_ids)+1
		    #print "class id is: "+class_id
                    if len(self.class_colors) > 0 and class_id in self.class_colors:
                        val = self.class_colors[class_id]
		    else:
        			val=0
		    val=class_id
                    id_image[mask == 1] = val
		    #id_image[mask == 1]=class_id	
                    #exported_class_ids.append(int(class_id))
                    exported_class_ids.append(val)
                    exported_rois.append(rois[m,:].tolist())
		    exported_scores.append(scores[m])
		    print type(scores[m])
		    print scores[m]
        global current_segmentation
        global current_class_ids
        global current_bounding_boxes
	global current_score
        current_segmentation=id_image
        current_class_ids=exported_class_ids
	current_score=exported_scores
	print exported_class_ids
        current_bounding_boxes=exported_rois
        return id_image, exported_class_ids, exported_rois, current_score


    def GetDynSeg(self,image,image2=None):
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
	results = self.model.detect([image], verbose=0)
	# Visualize results
	r = results[0]
	i = 0
	mask = np.zeros((h,w))
	for roi in r['rois']:
    		image_m = r['masks'][:,:,i]		
    		mask[image_m == 1] = self.class_colors[self.class_names[r['class_ids'][i]]]
		#print self.class_colors[self.class_names[r['class_ids'][i]]]
		'''
		if self.class_names[r['class_ids'][i]] == 'person':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'bicycle':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'car':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'motorcycle':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'airplane':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'bus':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'train':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'truck':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'boat':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'bird':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'cat':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'dog':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'horse':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'sheep':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'cow':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'elephant':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'bear':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'zebra':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.
		if self.class_names[r['class_ids'][i]] == 'giraffe':
			image_m = r['masks'][:,:,i]
			mask[image_m == 1] = 1.		
		'''
		i+=1
	#print('GetSeg mask shape:',mask.shape)
	return mask

    
	

    



import numpy as np
import tensorflow as tf

def preprocessLabels(filename):
	labelfile = open(filename, 'r')
	imageLabels = list()
	
	for line in labelfile:
		labels = line.split()
		labels = map(int, labels)
		imageLabels.append(labels)

	imageLabels = np.array(imageLabels)
	return imageLabels
		

imageLabels = preprocessLabels('iccv09Data/labels/0000047.regions.txt')	
print np.shape(imageLabels)	

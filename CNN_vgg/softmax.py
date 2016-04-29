import caffe
import pdb
import sys 
import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from tools import SimpleTransformer
import json
from copy import copy
def main():
	#caffe.set_mode_cpu()
	caffe.set_device(0)
	caffe.set_mode_gpu()
	directory = '.'	
	testNet = caffe.Net(osp.join(directory, 'softmax_deploy.prototxt'), osp.join(directory, 'VGG_ILSVRC_16_layers_softmax_30000.caffemodel'), caffe.TEST)
	testNames = open('../test_names.txt', 'r')
	testNames = testNames.readlines()
	testNames = [name[0: -1] for name in testNames]
	predications = {}
	labels = {}
	correct = 0
	for i in range(8574):
		out = testNet.forward()
		predications[testNames[i]] = str(out['result'][0][0])
		labels[testNames[i]] = str(out['label'][0])
		correct += (out['label'][0] in out['result'][0][0])
		print out['result'][0][0]
		#print out['label'][0], out['result'][0][0][0]
	print correct / 8574.0
	labels = json.dumps(labels)
	f = open('labels_name_softmax.json', 'w')
	f.write(labels)
	f.close()
	predictions = json.dumps(predications)
	f = open('predications_name_softmax.json', 'w')
	f.write(predictions)
	f.close()
if __name__ == '__main__':
	main()

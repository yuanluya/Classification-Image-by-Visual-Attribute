import caffe
import pdb
import sys 
import os
import numpy as np
import os.path as osp
from tools import SimpleTransformer
import json
from copy import copy
def main():
	if len(sys.argv) != 2:
		print "invalid argument, please enter mode [train|test]"
		return
	#caffe.set_mode_cpu()
	caffe.set_device(0)
	caffe.set_mode_gpu()
	directory = '.'
	if sys.argv[1] == 'train':
		solver = caffe.SGDSolver(osp.join(directory, 'solver_vgg.prototxt'))
		solver.net.copy_from('VGG_ILSVRC_16_layers_visual_attribute.caffemodel')
		solver.solve()
	if sys.argv[1] != "test" and argv[1] != "train":
		print "invalid argument, please enter mode [train|test]"
		return
	testNet = caffe.Net(osp.join(directory, 'deploy_vgg.prototxt'), osp.join(directory, 'VGG_ILSVRC_16_layers_visual_attribute.caffemodel'), caffe.TEST)
	testNames = open('../test_names.txt', 'r')
	testNames = testNames.readlines()
	testNames = [name[0: -1] for name in testNames]
	predications = {}
	labels = {}
	for i in range(8574):
		print i
		out = testNet.forward()
		for prediction in out['result']:
			predications[testNames[i]] = str(list(prediction))
		for label in out['label']:
			labels[testNames[i]] = str(list(label))
	labels = json.dumps(labels)
	f = open('labels_name.json', 'w')
	f.write(labels)
	f.close()
	predictions = json.dumps(predications)
	f = open('predications_name.json', 'w')
	f.write(predictions)
	f.close()
if __name__ == '__main__':
	main()

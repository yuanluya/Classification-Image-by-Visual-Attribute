import caffe
import sys 
import os
import subprocess
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
sys.path.append('/User/Desktop/caffe/examples/pycaffe/')
from tools import SimpleTransformer

from copy import copy
def main():
	caffe.set_mode_cpu()
	net = caffe.Net('train.prototxt', 'squeezenet_v1.0.caffemodel', caffe.TRAIN)
	solver = caffe.SGDSolver('solver.prototxt')
	solver.test_nets[0].share_with(solver.net)
	solver.step(1)
	net.save('trained.caffemodel')
	#test see output
	testNet = caffe.Net('deploy.prototxt', 'trained.caffemodel', caffe.TEST)
	out = testNet.forward()
	print out
	#print 'shape', out['score'].data.shape
	#print 'data', out['score'].data
	#print 'first', out['score'].data[0, :]
if __name__ == '__main__':
	main()
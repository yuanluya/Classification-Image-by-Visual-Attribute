import sys 
import os
import subprocess
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from copy import copy

# matplotlib inline
plt.rcParams['figure.figsize'] = (6, 6)

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
sys.path.append(caffe_root + 'python')
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

from caffe import layers as L, params as P # Shortcuts to define the net prototxt.

sys.path.append("pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append("pycaffe") # the tools file is in this folder

import tools #this contains some tools that we need

project_root = '/Users/yuanluyao/Desktop/UM/EECS_442/final_project/Classification-Image-by-Visual-Attribute'

caffe.set_mode_cpu()

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# main netspec wrapper
def caffenet_multilabel(data_layer_params, datalayer):
    # setup the python data layer 
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'multilabel_layer', layer = datalayer, 
                               ntop = 2, param_str=str(data_layer_params))

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.score = L.InnerProduct(n.drop7, num_output = data_layer_params['label_num'])
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
    
    return str(n.to_proto())

workdir = osp.join(project_root, 'CNN_1')
if not os.path.isdir(workdir):
	subprocess.call('mkdir ' + workdir, shell = True)
	

solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(workdir, "trainnet.prototxt"), testnet_prototxt_path = osp.join(workdir, "testnet.prototxt"))
solverprototxt.sp['display'] = "1"
solverprototxt.sp['base_lr'] = "0.0001"
path = osp.join(workdir, 'solver.prototxt')
solverprototxt.write(path)

# write train net.
with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
    # provide parameters to the data layer as a python dictionary. Easy as pie!
    data_layer_params = dict(batch_size = 128, im_shape = [256, 256], split = 'train', project_root = project_root, label_num = 99)
    f.write(caffenet_multilabel(data_layer_params, 'MultilabelDataLayerSync'))

# write test net.
with open(osp.join(workdir, 'testnet.prototxt'), 'w') as f:
    data_layer_params = dict(batch_size = 128, im_shape = [256, 256], split = 'test', project_root = project_root, label_num = 99)
    f.write(caffenet_multilabel(data_layer_params, 'MultilabelDataLayerSync'))

solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
solver.test_nets[0].share_with(solver.net)
solver.step(1)
'''
transformer = tools.SimpleTransformer()
plt.figure()
image = transformer.deprocess(copy(solver.net.blobs['data'].data[0, ...]))
print solver.net.blobs['label'].data[0, :]
plt.imshow(image)
plt.show()
'''

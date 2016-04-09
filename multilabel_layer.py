# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe


import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image
from tools import SimpleTransformer
import json


class MultilabelDataLayerSync(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a multilabel model on
    visa_dataset.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the paramameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        label_num = params['label_num']
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the 20 channels (because PASCAL has 20 classes.)
        top[1].reshape(self.batch_size, label_num)

        print_info("MultilabelDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, multilabel = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.root = params['project_root']
        #locate images data train/test_images
        self.images_direct = osp.join(self.root, params['split'] + '_images')
        #load image labels
        f = open(osp.join(self.root, 'labels.json'))
        labels = f.readlines()
        self.labels_dictionary = json.loads(labels[0])

        self.im_shape = params['im_shape']
        # get list of image indexes.
        self.namelist = os.listdir(self.images_direct)
        self.namelist = [name for name in self.namelist if name[name.rfind('.'): ] == '.JPEG']
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        print "BatchLoader initialized with {} images".format(
            len(self.namelist))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.namelist):
            self._cur = 0
            shuffle(self.namelist)

        # Load an image
        name = self.namelist[self._cur]  # Get the image index
        im = np.asarray(Image.open(
            osp.join(self.images_direct, name)))
        im = scipy.misc.imresize(im, self.im_shape)  # resize
        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2)*2-1
        im = im[:, ::flip, :]

        # Load and prepare ground truth
        multilabel = self.labels_dictionary[name[0: name.find('_')]]['features']

        self._cur += 1
        return self.transformer.preprocess(im), multilabel
        #return im, multilabel

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'project_root', 'im_shape', 'label_num']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])

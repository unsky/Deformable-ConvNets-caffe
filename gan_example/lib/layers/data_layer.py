# written by longriyao
#coding: utf-8
"""
    The data_layer is provide data to the net
"""
import caffe
import yaml
import os
import numpy as np
import cv2
import random
class DataLayer(caffe.Layer):

    _current = 0
    
    def _get_next_batch_paths(self):
        if self._current + self._batch_size >= len(self._image_paths):
            batch = self._image_paths[self._current:]
            
            more = self._batch_size - (len(self._image_paths) - self._current)
            batch.extend(self._image_paths[0:more])
            self._current = more
        else:
            batch = self._image_paths[self._current:(self._current + self._batch_size)]
            self._current += self._batch_size
        return batch

    def _get_batch(self,batch_paths):
        blobs = {}
        # store blob
        gray_image = np.zeros((self._batch_size,self._height,self._weight,2),dtype=np.float32)
        color_image = np.zeros((self._batch_size,self._height,self._weight,3),dtype=np.float32)

        for i in xrange(self._batch_size):
            im = cv2.imread(batch_paths[i],cv2.IMREAD_COLOR)
            im = im.astype(np.float32, copy=False)
            #im = np.array(im,dtype=np.float32,copy=False)
            # resize image according to height and weight
            if (self._height != im.shape[0] or self._weight != im.shape[1]):
                im = cv2.resize(im,(self._height,self._weight),interpolation = cv2.INTER_CUBIC)
            color_image[i,0:im.shape[0], 0:im.shape[1], :] = im
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            gray_image[i,0:im.shape[0], 0:im.shape[1], 0] = im
            #generate noise
            noise = np.random.uniform(0.0, 1.0, im.shape)
            gray_image[i, 0:im.shape[0], 0:im.shape[1], 1] = noise
        # change the dims like caffe
        channel_swap = (0, 3, 1, 2)
        color_image = color_image.transpose(channel_swap)
        gray_image = gray_image.transpose(channel_swap)

        #normalize the input
        color_image /= 255.0
        gray_image /= 255.0

        # generate noise
        #noise = np.random.uniform(0.0,1.0, gray_image.shape)
        #gray_image = gray_image + noise

        blobs['gray_image'] = gray_image
        blobs['color_image'] = color_image

        if self._has_label:
            label = np.zeros((len(batch_paths) * 2,1), dtype=np.int)
            label[:(len(batch_paths) - 1)] = 1
            blobs['label'] = label
        if self._has_label_g:
            label_g = np.zeros((len(batch_paths) * 2,1), dtype=np.int)
            label_g[(len(batch_paths) - 1):] = 1
            blobs['label_g'] = label_g
        return blobs

    def _get_next_batch(self):

        next_batch_paths = self._get_next_batch_paths()
        #batch_data_path = [self._image_paths[i] for i in indexs]
        return self._get_batch(next_batch_paths)
    
    def _get_param(self):
        # parse 
        layer_params = yaml.load(self.param_str)
        # get the param
        if layer_params.has_key('batch_size'):
            self._batch_size = layer_params['batch_size']
        else:
            self._batch_size = 32

        if layer_params.has_key('root_folder'):
            self._root_folder = layer_params['root_folder']
        else:
            self._root_folder = ""

        if layer_params.has_key('source'):
            self._source = layer_params['source']
        else:
            print "must write the source field in data_layer"
            assert 0
        if layer_params.has_key('height'):
            self._height = layer_params['height']
        else:
            self._height = 64

        if layer_params.has_key('weight'):
            self._weight = layer_params['weight']
        else:
            self._weight = 64
        if layer_params.has_key('shuffle'):
            self._shuffle = layer_params['shuffle']
        else:
            self._shuffle = True
            
    def _get_data_path(self):    
        file = open(self._source,'r')
        self._image_paths = file.readlines()
        
        if self._shuffle:
            random.shuffle(self._image_paths)
        # add folder 
        if self._root_folder:
            for i in xrange(len(self._image_paths)):
                self._image_paths[i] = os.path.join(self._root_folder,self._image_paths[i].strip('\n'))
                
    def setup(self,bottom,top):
        # setup function

        #map name to top index
        self._blob_name_to_index = {}
        # get param 
        self._get_param()
        #judce has label
        self._has_label = (len(top) >= 3)
        self._has_label_g = (len(top) >= 4)
        # get data path from param 
        self._get_data_path()
        # reshape top 
        top[0].reshape(self._batch_size, 2, self._height, self._weight)
        self._blob_name_to_index["gray_image"] = 0

        top[1].reshape(self._batch_size, 3, self._height, self._weight)
        self._blob_name_to_index["color_image"] = 1
        if self._has_label:
            self._blob_name_to_index["label"] = 2
            top[2].reshape(self._batch_size*2,1)
        if self._has_label_g:
            self._blob_name_to_index["label_g"] = 3
            top[3].reshape(self._batch_size*2,1)

    def forward(self,bottom,top):

        blobs = self._get_next_batch()

        for blob_name, blob in blobs.iteritems():
            top_index = self._blob_name_to_index[blob_name]
            #copy data to top
            top[top_index].data[...] = blob.astype(np.float32,copy=False)

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass

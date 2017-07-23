# written by longriyao
#coding: utf-8

import caffe
import cv2
import numpy as np
import yaml
import os
class GenLayer(caffe.Layer):
    def _get_param(self):
        # parse
        layer_params = yaml.load(self.param_str)
        # get the param
        if layer_params.has_key('output_folder'):
            self._output_folder = layer_params['output_folder']
        else:
            self._output_folder = ""

    def setup(self,bottom,top):
        self._get_param()
        self._i = 0
        top[0].reshape(1)
        pass

    def forward(self,bottom,top):
        if(bottom[0] is None):
            print "this is no images"
            return 0
        channel_swap = (0, 2, 3, 1)
        images = np.array(bottom[0].data)
        color_images = np.array(bottom[1].data)
        
        images = images.transpose(channel_swap)
        color_images = color_images.transpose(channel_swap)

        # get the origin image
        images *= 255.0
        color_images *= 255.0

        tmp_images = np.zeros((images[0].shape[0],images[0].shape[1]*2,images[0].shape[2]))
        
        # record the file num
        for i in xrange(len(images)):
            name = os.path.join(self._output_folder,str(self._i)+".jpg")
            tmp_images[:,0:images[i].shape[1],:] = color_images[i]
            tmp_images[:,images[i].shape[1]:,:] = images[i]
            
            cv2.imwrite(name,tmp_images)
            self._i = self._i + 1
        top[0].reshape(1)
        print "------------ forward is done"

    def backward(self,bottom,top):
        pass
    def reshape(self,bottom,top):
        pass

#!/usr/bin/env python

"""
Wrapper function for PVANet
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect, bbox_vote
from fast_rcnn.nms_wrapper import nms
import caffe, os, cv2
from utils.timer import Timer
import numpy as np

class PVANet:
    # for VOC07 test and evaluation
    MODEL_VOC07 = ('faster_rcnn_train_test_21cls.pt', 'PVA9.1_ImgNet_COCO_VOC0712.caffemodel')
    # for VOC12 test and evaluation
    MODEL_VOC12 = ('faster_rcnn_train_test_21cls.pt', 'PVA9.1_ImgNet_COCO_VOC0712plus.caffemodel')
    # a compressed version of the VOC12 model (slightly faster)
    MODEL_COMP = ('faster_rcnn_train_test_ft_rcnn_only_plus_comp.pt', 'PVA9.1_ImgNet_COCO_VOC0712plus_compressed.caffemodel')

    # Compute presets
    COMPUTE_FAST = {'bbox_vote': False, 'bbox_vote_n': 1, 'rpn_nms': 6000, 'proposals': 100}
    COMPUTE_BASE = {'bbox_vote': True, 'bbox_vote_n': 1, 'rpn_nms': 12000, 'proposals': 200}
    COMPUTE_FULL = {'bbox_vote': True, 'bbox_vote_n': 5, 'rpn_nms': 18000, 'proposals': 300}
    N_CLASSES = 20
    _net = None
    _im = None
    _thresh = 0.003

    def __init__(self, model=MODEL_VOC12, shorter=608, gpu_id=0, preset=COMPUTE_BASE):
        # Load cfg
        cfg.TEST.HAS_RPN = True
        cfg.TEST.SCALE_MULTIPLE_OF = 32
        cfg.TEST.MAX_SIZE = 2000
        cfg.TEST.SCALES = [shorter]
        cfg.TEST.BBOX_VOTE = preset['bbox_vote']
        cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE = preset['bbox_vote_n']
        cfg.TEST.BBOX_VOTE_WEIGHT_EMPTY = 0.3
        cfg.TEST.NMS = 0.4
        cfg.TEST.RPN_PRE_NMS_TOP_N = preset['rpn_nms']
        cfg.TEST.RPN_POST_NMS_TOP_N = preset['proposals']

        # Load model & pt
        pvanet_dir = os.path.abspath(os.path.join(cfg.ROOT_DIR, 'models', 'pvanet', 'pva9.1'))
        prototxt = os.path.join(pvanet_dir, model[0])
        caffemodel = os.path.join(pvanet_dir, model[1])

        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./models/pvanet/'
                           'download_all_models.sh?').format(caffemodel))

        if gpu_id is None:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
            cfg.GPU_ID = gpu_id

        net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        # Warm-up
        # Warmup on a dummy image
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _ = im_detect(net, im)

        self._net = net

    def read_img(self, filepath):
        self._im = cv2.imread(filepath)
        return self._im

    def process_img(self, im=None):
        if im is None:
            im = self._im

        scores, boxes = im_detect(self._net, im)

        # all detections are collected into:
        #    all_boxes[cls] = N x 5 array of detections in
        #    (i, x1, y1, x2, y2, score)
        all_boxes = np.zeros((0, 6))

        # skip j = 0, because it's the background class
        for j in xrange(1, PVANet.N_CLASSES):
            inds = np.where(scores[:, j] > self._thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)

            keep = nms(cls_dets, cfg.TEST.NMS)

            dets_NMSed = cls_dets[keep, :]
            if cfg.TEST.BBOX_VOTE:
                cls_dets = bbox_vote(dets_NMSed, cls_dets)
            else:
                cls_dets = dets_NMSed

            if len(cls_dets) > 0:
                all_boxes = np.vstack((all_boxes,
                                      np.hstack((np.ones((cls_dets.shape[0], 1)) * j,cls_dets))
                                       ))

        return all_boxes

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from demo import vis_detections

    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    # Init
    pvanet = PVANet(PVANet.MODEL_VOC07, preset=PVANet.COMPUTE_BASE)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)

        # Load the demo image
        im_file = os.path.join(cfg.DATA_DIR, 'demo', im_name)

        timer = Timer()
        timer.tic()
        im = pvanet.read_img(im_file)
        dets = pvanet.process_img(im)
        timer.toc()
        print ('Detection took {:.3f}s'.format(timer.total_time))

        # Visualize detections for each class
        CONF_THRESH = 0.7
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background

            det_this_cls = dets[np.where(dets[:,0] == cls_ind)[0]]
            boxes_and_scores = det_this_cls[:, 1:]

            if len(boxes_and_scores) > 0:
                vis_detections(im, cls, np.array(boxes_and_scores), thresh=CONF_THRESH)

    plt.show()
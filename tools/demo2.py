#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',  # always index 0
		   #  'aeroplane', 'bicycle', 'bird', 'boat',
		   #  'bottle', 'bus', 'car', 'cat', 'chair',
		   #  'cow', 'diningtable', 'dog', 'horse',
		   #  'motorbike', 'person', 'pottedplant',
		   #  'sheep', 'sofa', 'train', 'tvmonitor')
		   'rbc', 'wbc', 'platelets')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),
		'res101': ('res101_faster_rcnn_iter_10000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': (
	'voc_2007_trainval+voc_2012_trainval',)}


def vis_detections(im, class_name, dets, thresh=0.5):
	"""Draw detected bounding boxes."""
	inds = np.where(dets[:, -1] >= thresh)[0]
	if len(inds) == 0:
		return

	for i in inds:
		bbox = dets[i, :4]
		score = dets[i, -1]

		xmin = int(bbox[0])
		ymin = int(bbox[1])
		xmax = int(bbox[2])
		ymax = int(bbox[3])
		size = 1e-3 * im.shape[0]

		if class_name[0] == "w":
			cv2.rectangle(im, (xmin, ymin),
						  (xmax, ymax), (0, 0, 255), 1)
			cv2.putText(im, '{:s} {:.2f}'.format(class_name, score), (xmin + 10, ymin + 15),
						cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 255), 1)
		if class_name[0] == "r":
			cv2.rectangle(im, (xmin, ymin),
						  (xmax, ymax), (0, 255, 0), 1)
			cv2.putText(im, '{:s} {:.2f}'.format(class_name, score), (xmin + 10, ymin + 15),
						cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 1)
		if class_name[0] == "p":
			cv2.rectangle(im, (xmin, ymin),
						  (xmax, ymax), (255, 0, 0), 1)
			cv2.putText(im, '{:s} {:.2f}'.format(class_name, score), (xmin + 10, ymin + 15),
						cv2.FONT_HERSHEY_SIMPLEX, size, (255, 0, 0), 1)


def demo(sess, net, image_name):
	"""Detect object classes in an image using pre-computed object proposals."""

	# Load the demo image
	im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
	im = cv2.imread(im_file)

	# Detect all object classes and regress object bounds
	timer = Timer()
	timer.tic()
	scores, boxes = im_detect(sess, net, im)
	timer.toc()
	print('Detection took {:.3f}s for {:d} object proposals'.format(
		timer.total_time, boxes.shape[0]))

	# Visualize detections for each class
	CONF_THRESH = 0.8
	NMS_THRESH = 0.3

	# _, ax = plt.subplots(figsize=(12, 12))
	# im = im[:, :, (2, 1, 0)]
	# ax.imshow(im, aspect='equal')

	for cls_ind, cls in enumerate(CLASSES[1:]):
		cls_ind += 1  # because we skipped background
		cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes,
						  cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]
		vis_detections(im, cls, dets, thresh=CONF_THRESH)

	cv2.imwrite("results.png", im)

	# plt.axis('off')
	# # plt.tight_layout()
	# # plt.draw()
	# plt.savefig("./results.png")


def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(
		description='Tensorflow Faster R-CNN demo')
	parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
						choices=NETS.keys(), default='res101')
	parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
						choices=DATASETS.keys(), default='pascal_voc')
	args = parser.parse_args()

	return args


if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals
	args = parse_args()

	# model path
	demonet = args.demo_net
	dataset = args.dataset
	tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
						   NETS[demonet][0])

	if not os.path.isfile(tfmodel + '.meta'):
		raise IOError(('{:s} not found.\nDid you download the proper networks from '
					   'our server and place them properly?').format(tfmodel + '.meta'))

	# set config
	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True

	# init session
	sess = tf.Session(config=tfconfig)
	# load network
	if demonet == 'vgg16':
		net = vgg16()
	elif demonet == 'res101':
		net = resnetv1(num_layers=101)
	else:
		raise NotImplementedError
	net.create_architecture("TEST", 4,
							tag='default', anchor_scales=[8, 16, 32])
	saver = tf.train.Saver()
	saver.restore(sess, tfmodel)

	print('Loaded network {:s}'.format(tfmodel))

	im_names = ['raw_00360.jpg']
	for im_name in im_names:
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		print('Demo for data/demo/{}'.format(im_name))
		demo(sess, net, im_name)

	plt.show()

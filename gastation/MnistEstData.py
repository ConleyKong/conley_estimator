#-*- coding:utf-8 -*-
"""
-----------------------------------
    Project Name:    conley_estimator
    File Name   :    MnistEstData
    Author      :    Conley.K
    Create Date :    2020/4/13
    Description :    用于Estimator测试的数据载入,tf.data.Dataset interface to the MNIST dataset.
--------------------------------------------
    Change Activity: 
        2020/4/13 10:02 : 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import shutil
import warnings  #抑制numpy警告
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

basedir = os.path.abspath(os.path.dirname(__file__)) + "/"
sys.path.append(basedir)
warnings.simplefilter(action='ignore', category=FutureWarning)

class MnistDataset(object):
	def __init__(self,datadir):
		super(MnistDataset,self).__init__()
		self.datadir = datadir

	def read32(self,bytestream):
		"""Read 4 bytes from bytestream as an unsigned 32-bit integer."""
		dt = np.dtype(np.uint32).newbyteorder('>')
		return np.frombuffer(bytestream.read(4), dtype=dt)[0]


	def check_image_file_header(self,filename):
		"""Validate that filename corresponds to images for the MNIST dataset."""
		with tf.io.gfile.GFile(filename, 'rb') as f:
			magic = self.read32(f)
			self.read32(f)  # num_images, unused
			rows = self.read32(f)
			cols = self.read32(f)
			if magic != 2051:
				raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
				                                                               f.name))
			if rows != 28 or cols != 28:
				raise ValueError(
					'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
					(f.name, rows, cols))


	def check_labels_file_header(self,filename):
		"""Validate that filename corresponds to labels for the MNIST dataset."""
		with tf.io.gfile.GFile(filename, 'rb') as f:
			magic = self.read32(f)
			self.read32(f)  # num_items, unused
			if magic != 2049:
				raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
				                                                               f.name))


	def download(self,filename):
		"""Download (and unzip) a file from the MNIST dataset if not already done."""
		directory = self.datadir
		filepath = os.path.join(directory, filename)
		if tf.io.gfile.exists(filepath):
			return filepath
		if not tf.io.gfile.exists(directory):
			tf.gfile.MakeDirs(directory)
		# CVDF mirror of http://yann.lecun.com/exdb/mnist/
		url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
		_, zipped_filepath = tempfile.mkstemp(suffix='.gz')
		print('Downloading %s to %s' % (url, zipped_filepath))
		urllib.request.urlretrieve(url, zipped_filepath)
		with gzip.open(zipped_filepath, 'rb') as f_in, \
				tf.io.gfile.GFile(filepath, 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)
		os.remove(zipped_filepath)
		return filepath


	def dataset(self,images_file, labels_file):
		"""Download and parse MNIST dataset."""
		directory = self.datadir
		print(directory)
		images_file = self.download(images_file)
		labels_file = self.download(labels_file)

		self.check_image_file_header(images_file)
		self.check_labels_file_header(labels_file)

		def decode_image(image):
			# Normalize from [0, 255] to [0.0, 1.0]
			image = tf.decode_raw(image, tf.uint8)
			image = tf.cast(image, tf.float32)
			image = tf.reshape(image, [784])
			return image / 255.0

		def decode_label(label):
			label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
			label = tf.reshape(label, [])  # label is a scalar
			return tf.to_int32(label)

		images = tf.data.FixedLengthRecordDataset(images_file, 28 * 28, header_bytes=16).map(decode_image)
		labels = tf.data.FixedLengthRecordDataset(labels_file, 1, header_bytes=8).map(decode_label)
		return tf.data.Dataset.zip((images, labels))

	def onehot_map_func(self,images, label,label_num = 10):
		# images and label are tf.Tensor
		one_hot_label = tf.one_hot(label, depth=label_num)
		return images, one_hot_label


	def get_train_dataset(self,batch_size=12,epoch=None,
	                      shuffle_size=5000,prefetch_size=None):
		"""tf.data.Dataset object for MNIST training data."""
		dataset = self.dataset('train-images.idx3-ubyte','train-labels.idx1-ubyte').map(self.onehot_map_func)\
			.shuffle(shuffle_size).repeat(epoch).batch(batch_size)
		dataset = dataset.prefetch(None)
		return dataset


	def get_test_dataset(self,batch_size=32,epoch=1,
	                      shuffle_size=5000,prefetch_size=None):
		"""tf.data.Dataset object for MNIST test data."""
		dataset = self.dataset('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte').map(self.onehot_map_func) \
			.shuffle(shuffle_size).repeat(epoch).batch(batch_size)
		# dataset = dataset.prefetch(None)
		return dataset


def run():

	MNIST = MnistDataset(basedir+"/../data/mnist/")
	dataset = MNIST.get_train_dataset(batch_size=3)
	iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
	next_element = iterator.get_next()  # 词典，特征名到tensor的映射，无session时是tensor
	sess = tf.compat.v1.InteractiveSession()
	i = 1
	while True:
		try:
			itm = sess.run([next_element])[0]
		#
		# print(sess.run(net))

		except tf.errors.OutOfRangeError:
			print("End of dataset")
			break
		else:
			print('==============example %s ==============' % i)
			data, label = itm
			print("image:\n[{}]".format(data))
			print("label:\n[{}]".format(label))
			# print("----------\n total:\n{}".format(itm))

		i += 1
		if i > 2:
			break


if __name__ == "__main__":
	run()

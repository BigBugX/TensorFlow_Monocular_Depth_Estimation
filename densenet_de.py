#import torch
import collections
import numpy as np
import torch

import tensorflow as tf
import TensorflowUtils_plus as utils
from TensorflowUtils_plus import de_get_nk as get_nk

import loaddata
import matplotlib
import matplotlib.pyplot as plt
import scipy.misc as misc

import cv2

plt.set_cmap("gray")
# import pickle

"""
for layer, weights in vl:
	print(layer, weights.shape)
	layer_count = layer_count + 1

print("layer_count", layer_count)
"""

features = 'features/'
denseblock = 'denseblock'
denselayer = 'denselayer'
transition = 'transition'
conv0 = 'conv0/'
conv1 = 'conv1/'
conv2 = 'conv2/'
norm0 = 'norm0/'
norm1 = 'norm1/'
norm2 = 'norm2/'
sub_l = ['weight', 'bias', 'running_mean', 'running_var']
denseparams = [6, 12, 32, 32]
#block_outputs = []
batch_size = 1


def dense_de_block(weights, feature_maps, block_idx, block_outputs):
	with tf.variable_scope("denseblock%d" % block_idx):
		global output
		tran_input = dense_de_layer(weights, feature_maps, block_idx)
		if (block_idx < 4):
			output = utils.tran_de(weights, tran_input, "tran%d" % block_idx)
			block_outputs.append(output)
		if (block_idx == 4):
			block_outputs.append(tran_input)
		#return output, block_outputs
		return block_outputs


def dense_de_layer(weights, feature_maps, block_idx,):
	layer_num = denseparams[block_idx-1]
	block_net = []
	block_net.append(feature_maps)
	"""
	layer_input = feature_maps
	for i in range(1, layer_num+1):
		for j in range(1, i+1):
			if (j == 1):
				continue
			else:
				layer_input = tf.concat([layer_input, block_net[j]], 3)
		res = dense_de_sub_layer(weights, layer_input, block_idx, i)
		block_net.append(res)
	"""
	for i in range(1, layer_num+1):
		for j in range(1, i+1):
			if (j == 1):
				layer_input = feature_maps
			else:
				layer_input = tf.concat([layer_input, block_net[j-1]], 3)
		res = dense_de_sub_layer(weights, layer_input, block_idx, i)
		block_net.append(res)
	
	res = feature_maps
	for i in range(1, len(block_net)):
		res = tf.concat([res, block_net[i]], 3)
	return res


def dense_de_sub_layer(weights, feature_maps, block_idx, layer_idx):
	with tf.variable_scope("denselayer%d" % layer_idx):
		l_scope = features + denseblock + (str)(block_idx) + '/' + denselayer + (str)(layer_idx) + '/'
		nk = get_nk(weights, block_idx, layer_idx)
		c1k = weights[l_scope + conv1 + sub_l[0]]
		c2k = weights[l_scope + conv2 + sub_l[0]]
		res = utils.batch_norm_de(feature_maps, nk[2], nk[3], nk[1], nk[0], scope='bn1')
		res = utils.relu_de(res, name='relu1')
		res = utils.conv2d_de(res, c1k, 1, 1, name='conv1')
		res = utils.batch_norm_de(res, nk[6], nk[7], nk[5], nk[4], scope='bn2')
		res = utils.relu_de(res, name='relu2')
		res = utils.conv2d_de(res, c2k, 3, 1, name='conv2')
	return res


def dense_de(weights, refine_weights, image):
	# define and hold the char for the dense_de model
	block_outputs = []
	current = image
	nk = [] # norm kernel list

	## adding for resize the image
	## org_shape = image.get_shape().as_list()
	## _, org_height, org_width, channels = org_shape


	with tf.variable_scope("inference"):
		# add structure into the model
		# first layer(s)
		kernels = weights[features + conv0 + sub_l[0]]
		current = utils.conv2d_de(current, kernels, 7, stride=2, name='conv0')
		nk = get_nk(weights, 0, 0)
		current = utils.batch_norm_de(current, nk[2], nk[3], nk[1], nk[0], scope='bn0')
		current = utils.relu_de(current, name='relu0')
		current = utils.maxpool_de(current, pool_size=3, stride=2, name='pool0')
		block_outputs.append(current)	

		# add 4 blocks with params [6, 12, 32, 32]
		for i in range(1,5):
			block_outputs = dense_de_block(weights, current, i, block_outputs)
			current = block_outputs[-1]	

		# add norm5 and conv1(which is 'conv2' in the original paper)
		# 9 layer totally
		layer_norm5 = features + 'norm5/'
		with tf.variable_scope("norm5"):
			n5_w = weights[layer_norm5 + sub_l[0]]
			n5_b = weights[layer_norm5 + sub_l[1]]
			n5_m = weights[layer_norm5 + sub_l[2]]
			n5_v = weights[layer_norm5 + sub_l[3]]
			conv1_kernels = weights['conv1/weight']
			current = utils.batch_norm_de(current, n5_m, n5_v, n5_b, n5_w, scope='bn')
		current = utils.conv2d_de(current, conv1_kernels, 1, 1, name='conv1')
					# batchnorm of norm5
		layer_bn = 'bn/'
		bn_w = weights[layer_bn + sub_l[0]]
		bn_b = weights[layer_bn + sub_l[1]]
		bn_m = weights[layer_bn + sub_l[2]]
		bn_v = weights[layer_bn + sub_l[3]]
		current = utils.batch_norm_de(current, bn_m, bn_v, bn_b, bn_w, scope='bn')
		block_outputs.append(current)	

		# add 4 upsampling blocks
		for i in range(1,5):
			current = utils.up_block_de(block_outputs[-1], weights, i)
			block_outputs.append(current)	

		# add conv2(which is 'conv3' in the original paper)
		conv2_w = weights['conv2/weight']
		conv2_w = conv2_w.transpose((2,3,1,0))
		conv2_b = weights['conv2/bias']
		current = utils.conv2d_bias_de(current, conv2_w, kernel_size=3, stride=1, name='conv2', padding=1, bias=conv2_b)
		block_outputs.append(current) 
		# the ouput of the base net is with idx 10

		refine_block_inputs = []
		refine_block_outputs = []
		for i in range(1, 5):
			refine_block_inputs.append(block_outputs[i])
		refine_shape = block_outputs[-1].get_shape().as_list()

		with tf.variable_scope("refine"):
			for i in range(1,5):
				current = utils.up_block_refine_de(refine_block_inputs[i-1], refine_weights, refine_shape[2], refine_shape[1], i)
				refine_block_outputs.append(current)
			#conv1_input = refine_block_outputs[0]
			#for i in range(1,4):
			#	conv1_input = tf.concat([conv1_input, refine_block_outputs[i]], 3)
			conv1_input = tf.concat([refine_block_outputs[0],refine_block_outputs[1],refine_block_outputs[2],refine_block_outputs[3]],3)
			
			conv1_w = refine_weights['conv1/weight']
			rbn_m = refine_weights['bn1/running_mean']
			rbn_v = refine_weights['bn1/running_var']
			rbn_b = refine_weights['bn1/bias']
			rbn_w = refine_weights['bn1/weight']
			res = utils.conv2d_de(conv1_input, conv1_w, 5, 1, name='conv1', padding="SAME")
			res = utils.batch_norm_de(res, rbn_m, rbn_v, rbn_b, rbn_w, scope='bn1')
			res = utils.relu_de(res, name='relu1')
			refine_block_outputs.append(res)

			res = tf.concat([res, block_outputs[-1]], 3)

			conv2_w = refine_weights['conv2/weight']
			rbn_m = refine_weights['bn2/running_mean']
			rbn_v = refine_weights['bn2/running_var']
			rbn_b = refine_weights['bn2/bias']
			rbn_w = refine_weights['bn2/weight']
			res = utils.conv2d_de(res, conv2_w, 5, 1, name='conv2', padding="SAME")
			res = utils.batch_norm_de(res, rbn_m, rbn_v, rbn_b, rbn_w, scope='bn2')
			res = utils.relu_de(res, name='relu2')
			refine_block_outputs.append(res)

			conv3_w = refine_weights['conv3/weight']
			rbn_m = refine_weights['bn3/running_mean']
			rbn_v = refine_weights['bn3/running_var']
			rbn_b = refine_weights['bn3/bias']
			rbn_w = refine_weights['bn3/weight']
			res = utils.conv2d_de(res, conv3_w, 5, 1, name='conv3', padding="SAME")
			res = utils.batch_norm_de(res, rbn_m, rbn_v, rbn_b, rbn_w, scope='bn3')
			res = utils.relu_de(res, name='relu3')
			refine_block_outputs.append(res)

			conv4_w = refine_weights['conv4/weight']
			conv4_w = conv4_w.transpose((2,3,1,0))
			conv4_b = refine_weights['conv4/bias']
			res = utils.conv2d_bias_de(res, conv4_w, kernel_size=5, stride=1, name='prediction', padding=1, bias=conv4_b)
			refine_block_outputs.append(res)
			## block_outputs.append(res)
			## res = misc.imresize(res,(org_height, org_width))

		#return block_outputs
		return res
	
"""
def inference(image):
	# fill the dense_de model with pretrained weights
"""

def pred():
	# define the session of inference, and feed data into it
	image = tf.placeholder(tf.float32, shape=[batch_size, 228, 304, 3], name='input_image')
	
	# get weights of base_net
	model_pt = torch.load('models\\base_nyu2_tf', map_location='cpu')
	weights = {} 
	for layer, w in model_pt.items():
		weights[layer] = w.cpu().detach().numpy()

	# get weights of refine_net
	model_pt = torch.load('models\\refine_nyu2_tf', map_location='cpu')
	refine_weights = {}
	for layer, w in model_pt.items():
		refine_weights[layer] = w


	pred_annotation = dense_de(weights, refine_weights, image)
	data_loader = loaddata.readNyu2('data/resized.jpg')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i, imstream in enumerate(data_loader):
			imstream = imstream.numpy()
			imstream = imstream.transpose((0,2,3,1))
			feed_dict = {image: imstream}
			preds = sess.run(pred_annotation, feed_dict=feed_dict)

			#outputimg = preds[-1]
			outputimg = preds
			outputimg = outputimg.transpose((0,3,1,2))
			outputimg_ts = torch.from_numpy(outputimg)
			matplotlib.image.imsave("data\\mpl_im.png", outputimg_ts.view([114,152]))
			misc.imsave("data\\misc_im1003.png", outputimg_ts.view([114,152]))
			cv2.imwrite("data\\cv2_im1003.png", outputimg_ts.view([114,152]).cpu().detach().numpy())


		#saving the graph
		LOGDIR = "data"
		writer = tf.summary.FileWriter(LOGDIR)
		writer.add_graph(sess.graph)

if __name__ == '__main__':
	pred()

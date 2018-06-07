__author__ = 'jason'

import tensorflow as tf 
import sys
from tool import NUM_PARTS

INPUT_CHANNEL = 1
CONV1_DEEP = 16
CONV1_SIZE = 7
CONV2_DEEP = 64
CONV2_SIZE = 7
CONV3_DEEP = 256
CONV3_SIZE = 7
DROP_RATE = 0.5
WEIGHT_DECAY=0.0005

def weightVariable(shape, weight_decay):
	init = tf.Variable(tf.random_normal(shape, stddev=0.01))
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(weight_decay)(init))
	return init

def biasVariable(shape):
	init = tf.random_normal(shape)
	return tf.Variable(init)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

def normalize(x):
	return tf.nn.local_response_normalization(x, 5/2, 2, 1e-4, 0.75)

def dropout(x, keep):
	return tf.nn.dropout(x, keep)

def full_inference(net, probe_gei, gallery_gei):
	if net=="MT":
		return MT_inference(probe_gei, gallery_gei)
	else:
		print("Error: Wrong net type")
		sys.exit(1)

def partial_inference(net, parted_probe_gei, parted_gallery_gei):
	if net=="TMT":
		return TMT_inference(parted_probe_gei, parted_gallery_gei)	
	elif net=="BMT":
		return BMT_inference(parted_probe_gei, parted_gallery_gei)
	else:
		print("Error: Wrong net type")
		sys.exit(1)

def part_MT_inference(part_gei0, part_gei1):
	W11 = weightVariable([CONV1_SIZE,CONV1_SIZE,INPUT_CHANNEL,CONV1_DEEP],WEIGHT_DECAY)
	b11 = biasVariable([CONV1_DEEP])
	conv11 = normalize(tf.nn.relu(conv2d(part_gei0,W11)+b11))
	pool11 = maxPool(conv11)
	
	W12 = weightVariable([CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],WEIGHT_DECAY)
	b12 = biasVariable([CONV2_DEEP])
	conv12 = normalize(tf.nn.relu(conv2d(pool11,W12)+b12))
	pool12 = maxPool(conv12)
	
	W21 = weightVariable([CONV1_SIZE,CONV1_SIZE,INPUT_CHANNEL,CONV1_DEEP],WEIGHT_DECAY)
	b21 = biasVariable([CONV1_DEEP])
	conv21 = normalize(tf.nn.relu(conv2d(part_gei1,W21)+b21))
	pool21 = maxPool(conv21)
	
	W22 = weightVariable([CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],WEIGHT_DECAY)
	b22 = biasVariable([CONV2_DEEP])
	conv22 = normalize(tf.nn.relu(conv2d(pool21,W22)+b22))
	pool22 = maxPool(conv22)

	W3 = weightVariable([CONV3_SIZE,CONV3_SIZE,CONV2_DEEP,CONV3_DEEP],WEIGHT_DECAY)
	b3 = biasVariable([CONV3_DEEP])
	conv3 = tf.nn.relu(conv2d(pool12,W3)+conv2d(pool22,W3)+b3)
	feature_map = dropout(conv3, DROP_RATE)
	#print feature_map.shape
	return feature_map

def classify(feature_map):
	#print("feature_map.shape:", feature_map.shape)
	tmp = int(feature_map.shape[1]*feature_map.shape[2]*feature_map.shape[3])	
	Wf = weightVariable([tmp,2],WEIGHT_DECAY)
	bf = biasVariable([2])
	flatted_map = tf.reshape(feature_map, [-1, tmp])
	out = tf.add(tf.matmul(flatted_map, Wf), bf)
	return out
	
def concat_maps(maps):
	#height=sum([int(maps[i].shape[1]) for i in [0,1,2,4,5]])
	#width=maps[0].shape[2]
	#print "height, width:", height, width
	tmp_map1=tf.concat([maps[0], maps[1]], 1)
	tmp_map2=tf.concat([maps[2], maps[3]], 2)
	tmp_map3=tf.concat([maps[4], maps[5]], 1)
	feature_map=tf.concat([tmp_map1, tmp_map2, tmp_map3], 1)
	#print "feature_map.shape:", feature_map.shape
	return feature_map

def MT_inference(gei0, gei1):
	return part_MT_inference(gei0, gei1)

def TMT_inference(gei0, gei1):
	maps=[]
	for i in range(len(gei0)):
		maps.append(part_MT_inference(gei0[i], gei1[i]))	
	feature_map = concat_maps(maps)
	return feature_map

def BMT_inference(gei0, gei1):
	W11=[]
	b11=[]
	conv11=[]
	pool11=[]
	W21=[]
	b21=[]
	conv21=[]
	pool21=[]
	
	for i in range(len(gei0)):
		W11.append(weightVariable([CONV1_SIZE,CONV1_SIZE,INPUT_CHANNEL,CONV1_DEEP],WEIGHT_DECAY))
		b11.append(biasVariable([CONV1_DEEP]))
		conv11.append(normalize(tf.nn.relu(conv2d(gei0[i],W11[i])+b11[i])))
		pool11.append(maxPool(conv11[i]))
	for i in range(len(gei1)):
		W21.append(weightVariable([CONV1_SIZE,CONV1_SIZE,INPUT_CHANNEL,CONV1_DEEP],WEIGHT_DECAY))
		b21.append(biasVariable([CONV1_DEEP]))
		conv21.append(normalize(tf.nn.relu(conv2d(gei1[i],W21[i])+b21[i])))
		pool21.append(maxPool(conv21[i]))
	concated_pool11=concat_maps(pool11)	
	concated_pool21=concat_maps(pool21)	

	W12 = weightVariable([CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],WEIGHT_DECAY)
	b12 = biasVariable([CONV2_DEEP])
	conv12 = normalize(tf.nn.relu(conv2d(concated_pool11,W12)+b12))
	pool12 = maxPool(conv12)

	W22 = weightVariable([CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],WEIGHT_DECAY)
	b22 = biasVariable([CONV2_DEEP])
	conv22 = normalize(tf.nn.relu(conv2d(concated_pool21,W22)+b22))
	pool22 = maxPool(conv22)

	W3 = weightVariable([CONV3_SIZE,CONV3_SIZE,CONV2_DEEP,CONV3_DEEP],WEIGHT_DECAY)
	b3 = biasVariable([CONV3_DEEP])
	conv3 = tf.nn.relu(conv2d(pool12,W3)+conv2d(pool22,W3)+b3)
	feature_map = dropout(conv3, DROP_RATE)

	return feature_map
		
def LB_inference(gei0, gei1):
	W1 = weightVariable([CONV1_SIZE,CONV1_SIZE,INPUT_CHANNEL,CONV1_DEEP],WEIGHT_DECAY)
	b1 = biasVariable([CONV1_DEEP])
	conv1 = normalize(tf.nn.relu(conv2d(gei0,W1)+conv2d(gei1,W1)+b1))
	pool1 = maxPool(conv1)

	W2 = weightVariable([CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],WEIGHT_DECAY)
	b2 = biasVariable([CONV2_DEEP])
	conv2 = normalize(tf.nn.relu(conv2d(pool1,W2)+b2))
	pool2 = maxPool(conv2)
	
	W3 = weightVariable([CONV3_SIZE,CONV3_SIZE,CONV2_DEEP,CONV3_DEEP],WEIGHT_DECAY)
	b3 = biasVariable([CONV3_DEEP])
	conv3 = tf.nn.relu(conv2d(pool2,W3)+b3)
	drop3 = dropout(conv3, DROP_RATE)
	
	Wf = weightVariable([11*21*CONV3_DEEP,2],WEIGHT_DECAY)
	bf = biasVariable([2])
	drop3_flat = tf.reshape(drop3, [-1, 11*21*CONV3_DEEP])
	out = tf.add(tf.matmul(drop3_flat, Wf), bf)

	return out

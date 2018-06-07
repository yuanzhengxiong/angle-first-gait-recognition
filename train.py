__author__ = 'jason'

from parse_args import parse_arguments
from data_tool import load_data
from data_tool import prepare_training_data
from data_tool import get_next_batch
from data_tool import load_angle_train_data
from model.models import RandomForestClassification
from sklearn.externals import joblib
import os
import sys
from inference import classify
from inference import full_inference, partial_inference
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
from tool import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_PARTS
from tool import PARTED_HEIGHT, PARTED_WIDTH
from tool import segment_batch_img


BATCH_SIZE = 128
view_list= ["%03d" % (18*i) for i in range(0,11)]

def get_probe_gallery_dir(probe_type):
	if probe_type=="NM":
		probe_dir=["nm-05", "nm-06"]
	elif probe_type=="CL":
		probe_dir=["cl-01", "cl-02"]
	elif probe_type=="BG":
		probe_dir=["bg-01", "bg-02"]
	else:
		print("Error: wrong probe type.")
	gallery_dir = ["nm-01","nm-02","nm-03","nm-04"]
	return probe_dir, gallery_dir

def get_rfc_path(img_class, probe_type):
	rfc_path = "./ckpts/%s/RFC/rfc_%s.model" % (img_class, probe_type)
	return rfc_path

def get_ckpt_file(img_class, net, probe_type):
	if net=="MT":
		ckpt = tf.train.get_checkpoint_state('./ckpts/%s/full-MT-%s/' % (img_class, probe_type))
	elif net=="BMT":
		ckpt = tf.train.get_checkpoint_state('./ckpts/%s/partial-BMT-%s/' % (img_class, probe_type))
	elif net=="TMT":
		ckpt = tf.train.get_checkpoint_state('./ckpts/%s/partial-TMT-%s/' % (img_class, probe_type))
	else:
		print("Error: Wrong net type")
		sys.exit(1)
	return ckpt
	
def get_ckpt_filename_to_save(img_class, net, probe_type):
	if net=="MT":
		filename='./ckpts/%s/full-MT-%s/cnn-%s.ckpt' % (img_class, probe_type,probe_type)			
	elif net=="BMT":
		filename='./ckpts/%s/partial-BMT-%s/cnn-%s.ckpt' % (img_class, probe_type,probe_type)
	elif net=="TMT":
		filename='./ckpts/%s/partial-TMT-%s/cnn-%s.ckpt' % (img_class, probe_type,probe_type)			
	else:
		print("Error: Wrong net type")
		sys.exit(1)
	return filename

def get_feature_map(img_type, net):
	global probe_img, gallery_img
	global parted_probe_img,parted_gallery_img
	if img_type=="full":
		probe_img=tf.placeholder(tf.float32, [None,IMAGE_HEIGHT,IMAGE_WIDTH,1])
		gallery_img=tf.placeholder(tf.float32, [None,IMAGE_HEIGHT,IMAGE_WIDTH,1])
		feature_map = full_inference(net, probe_img, gallery_img)
	elif img_type=="partial":
		parted_probe_img=[tf.placeholder(tf.float32, [None,PARTED_HEIGHT[i],PARTED_WIDTH[i],1]) for i in range(0,6)]
		parted_gallery_img=[tf.placeholder(tf.float32, [None,PARTED_HEIGHT[i],PARTED_WIDTH[i],1]) for i in range(0,6)]
		feature_map = partial_inference(net, parted_probe_img, parted_gallery_img)
	elif img_type=="combined":
		pass
	else:
		print("Error: Wrong net type")	
		sys.exit(1)
	return feature_map
		
def train():
	### parse arguments
	img_class, gpu, img_type, net, probe_type, model = parse_arguments(sys.argv[1:])

	os.environ["CUDA_VISIBLE_DEVICES"] = '2'

	### get probe and gallery dir
	probe_dir, gallery_dir=get_probe_gallery_dir(probe_type)

	### angle train
	rfc_path=get_rfc_path(img_class, probe_type)
	if not os.path.exists(rfc_path):
		print("rfc model does not exist, need to train")
		rfc_model=RandomForestClassification()
		train_x, train_y=load_angle_train_data(img_class, view_list,probe_dir)	
		rfc=rfc_model.fit(x_train=train_x,y_train=train_y)
		joblib.dump(rfc, rfc_path)

	### cnn train
	y_ = tf.placeholder(tf.float32, [None, 2])
	y = classify(get_feature_map(img_type, net))
			
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
		logits = y, labels = y_))
	total_loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_, 1)), tf.float32))
	train_step = tf.train.AdamOptimizer(0.0001).minimize(total_loss)

	#gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
	gpu_options=tf.GPUOptions(allow_growth=True)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		saver = tf.train.Saver()
		paired_train_data = prepare_training_data(probe_dir, gallery_dir)
		NUM_BATCH = len(paired_train_data)//BATCH_SIZE
		sess.run(tf.global_variables_initializer())

		x_axis=[]
		y_axis=[]
		y1_axis=[]

		start_step=0
		ckpt = get_ckpt_file(img_class, net, probe_type)
		if ckpt and ckpt.model_checkpoint_path:
			print(ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
			start_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

		n = 0
		while(True):
			for i in range(NUM_BATCH):
				batch_x, batch_y = get_next_batch(img_class, probe_type, paired_train_data, BATCH_SIZE)
				batch_x0 = batch_x[:,0,:,:].reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH,1])	
				batch_x1 = batch_x[:,1,:,:].reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH,1])	
				if img_type=="full":
					_,loss=sess.run([train_step, total_loss],
						feed_dict={probe_img:batch_x0,gallery_img:batch_x1,y_:batch_y})
				elif img_type=="partial":
					parted_probe_img_list=segment_batch_img(batch_x0)
					parted_gallery_img_list=segment_batch_img(batch_x1)
					_,loss=sess.run([train_step, total_loss],
						feed_dict={y_:batch_y,
								   parted_probe_img[0]:parted_probe_img_list[0],
	   							   parted_probe_img[1]:parted_probe_img_list[1],
	   							   parted_probe_img[2]:parted_probe_img_list[2],
	   							   parted_probe_img[3]:parted_probe_img_list[3],
	   							   parted_probe_img[4]:parted_probe_img_list[4],
	   							   parted_probe_img[5]:parted_probe_img_list[5],
							   	   parted_gallery_img[0]:parted_gallery_img_list[0],
							   	   parted_gallery_img[1]:parted_gallery_img_list[1],
							   	   parted_gallery_img[2]:parted_gallery_img_list[2],
							   	   parted_gallery_img[3]:parted_gallery_img_list[3],
							   	   parted_gallery_img[4]:parted_gallery_img_list[4],
							   	   parted_gallery_img[5]:parted_gallery_img_list[5]
						       	   })
				elif img_type=="combined":
					pass
				else:
					print("Error: Wrong net type")
					sys.exit(1)

				global_step = int(start_step)+n*NUM_BATCH+i
				if (global_step % 10) == 0:
					print(global_step, " loss: ", loss)
				if (global_step % 100) == 0:
					accs=[]
					for view in view_list:
						val_x, val_y = load_data(img_class, "validation", view, probe_dir, gallery_dir)
						img0=val_x[:,0].reshape([-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])	
						img1=val_x[:,1].reshape([-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])	
						if img_type=="partial":
							parted_img0 = segment_batch_img(img0)	
							parted_img1 = segment_batch_img(img1)	
							acc = accuracy.eval({
 								parted_probe_img[0]:parted_img0[0],
 								parted_probe_img[1]:parted_img0[1],
 								parted_probe_img[2]:parted_img0[2],
 								parted_probe_img[3]:parted_img0[3],
 								parted_probe_img[4]:parted_img0[4],
 								parted_probe_img[5]:parted_img0[5],
								parted_gallery_img[0]:parted_img1[0],
								parted_gallery_img[1]:parted_img1[1],
								parted_gallery_img[2]:parted_img1[2],
								parted_gallery_img[3]:parted_img1[3],
								parted_gallery_img[4]:parted_img1[4],
								parted_gallery_img[5]:parted_img1[5],
								y_:val_y})
						elif img_type=="full":
							acc = accuracy.eval({probe_img:img0,gallery_img:img1,y_:val_y})
						elif img_type=="combined":
							pass
						else:
							print("Error: Wrong net type")
							sys.exit(1)
						accs.append(acc)
					avg_acc = sum(accs)/len(accs)
					x_axis.append(global_step) 
					y_axis.append(avg_acc) 
					y1_axis.append(loss)
					print(global_step, " accuracy: ", avg_acc)
					#if acc > 0.995 and n > 2:
					#	saver.save(sess, './chpts/cnn_MT.model', \
					#		global_step=n*NUM_BATCH+i)
					#	plt.plot(x_axis, y_axis)				
					#	plt.show()
					#	sys.exit(0)
				if global_step != 0 and (global_step % 1000) == 0:
					saver.save(sess, get_ckpt_filename_to_save(img_class, net, probe_type),\
						global_step=int(start_step)+n*NUM_BATCH+i)
				if global_step >= (int(start_step)+25000):
					plt.figure(1)	
					ax1 = plt.subplot(111)	
					ax2 = plt.subplot(112)	
					plt.sca(ax1)
					plt.plot(x_axis, y_axis, 'r', linewidth=2)
					plt.sca(ax2)
					plt.plot(x_axis, y1_axis, 'g', linewidth=2)
					plt.show()
					sys.exit(0)
			n += 1

if __name__ == "__main__":
	train()

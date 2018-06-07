import tensorflow as tf
import numpy as np
import os
import sys
from tool import IMAGE_HEIGHT, IMAGE_WIDTH
from tool import PARTED_HEIGHT, PARTED_WIDTH
from tool import segment_batch_img
from parse_args import parse_arguments 
from data_tool import load_data
from inference import classify
from inference import full_inference, partial_inference
from train import get_probe_gallery_dir
from train import get_ckpt_file
import matplotlib.pyplot as plt


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

def test():
	img_class, gpu, img_type, net, probe_type, model = parse_arguments(sys.argv[1:])	

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	probe_dir, gallery_dir = get_probe_gallery_dir(probe_type)
		
	y_ = tf.placeholder(tf.float32, [None, 2])
	y = classify(get_feature_map(img_type, net))
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1), \
		tf.argmax(y_,1)), tf.float32))
	
	gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)	
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		saver = tf.train.Saver()
		ckpt = get_ckpt_file(img_class, net, probe_type)
		if ckpt and ckpt.model_checkpoint_path:
			print(ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print("No checkpoint file found")
			return

		accs = []
		test_view_list = ["000","018","036","054","072","090","108","126","144","162","180"]
		for view in test_view_list:
			test_x, test_y = load_data(img_class, "testing", view, probe_dir, gallery_dir)
			img0 = test_x[:,0].reshape([-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])
			img1 = test_x[:,1].reshape([-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])
			if img_type=="full":
				acc = sess.run(accuracy, feed_dict={probe_img:img0,\
					gallery_img:img1, y_:test_y})	
			elif img_type=="partial":
				parted_probe_img_list=segment_batch_img(img0)	
				parted_gallery_img_list=segment_batch_img(img1)	
				acc = sess.run(accuracy, 
					feed_dict={parted_probe_img[0]:parted_probe_img_list[0],
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
						   	   parted_gallery_img[5]:parted_gallery_img_list[5],
					       	   y_:test_y})
			elif img_type=="combined":
				pass
			else:
				print("Error: Wrong net type")
				sys.exit(1)
			accs.append(acc)
			print('After %s training steps, probe view:' % global_step, view,\
				 'accuracy = %g' % acc)					

		average_acc = sum(accs)/len(accs)
		print('After %s training steps,' % global_step,
			 'average accuracy = %g' % average_acc)					

		x_axis = test_view_list
		y_axis = [100*x for x in accs]
		plt.plot(x_axis, y_axis)
		plt.plot(x_axis, y_axis, 'ro')
		plt.xlabel('Probe view angle')
		plt.ylabel('Verification accuracy')
		if probe_type=='NM':
			plt.title('Gallery: NM #1-4, view angles: 0-180 Probe: NM #5-6')
		elif probe_type=='BG':
			plt.title('Gallery: NM #1-4, view angles: 0-180 Probe: BG #1-2')
		elif probe_type=='CL':
			plt.title('Gallery: NM #1-4, view angles: 0-180 Probe: CL #1-2')
		else:
			print('Wrong probe type')
			
		plt.ylim(20, 100)
		plt.show()

if __name__ == "__main__":
	test()

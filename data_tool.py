import config
import sys
import tool
from random import sample
import csv
from skimage.io import imsave
from scipy import misc
from tool import img_path_to_IMG
import logging
import random
import numpy as np
import os
from feature.hog import flatten
from sklearn.externals import joblib

logger = logging.getLogger("data")

def get_angle(img_class, probe_type, imgs):
	model=joblib.load('./ckpts/%s/RFC/rfc_%s.model' % (img_class, probe_type))
	return model.predict(imgs)

def load_angle_train_data(img_class, view_list, train_dir):
	human_id_list=["%03d" % i for i in range(1, 51)]
	human_id_list.remove('005')
	human_id_list.remove('026') #126
	human_id_list.remove('034')
	human_id_list.remove('037') #144
	human_id_list.remove('046')
	human_id_list.remove('048') #54
	
	if view_list is None:
		view_list=["000","018","036","054","072","090","108","126","144","162","180"]
	if train_dir is None:
		train_dir = ["nm-%02d" % i for i in range(1, 5)]

	training_x=[]
	training_y=[]
	
	# check dir exists
	for id in human_id_list:
		for dir in train_dir:
			for view in view_list:
				img_dir = "%s/%s/%s/%s" % (config.Project.casia_dataset_b_path, id, dir, view)
				if not os.path.exists(img_dir):
					logger.error("%s do not exist" % img_dir)

	for id in human_id_list:
		logger.info("processing human %s" % id)
		for dir in train_dir:
			for view in view_list:
				img_dir="%s/%s/%s/%s" % (config.Project.casia_dataset_b_path,id,dir,view)
				data=img_path_to_IMG(img_class, img_dir)
				if len(data.shape) > 0:
					training_x.append(flatten(data))
					training_y.append(view)
				else:
					print("LOAD_ANGLE_TRAIN_DATA: fail to extract %s of %s" % (img_dir, id))
					sys.exit(0)
	
	return training_x, training_y

def check_dir(human_id_list, dir_list, view_list):
	for id in human_id_list:
		for dir in dir_list:
			for view in view_list:
				img_dir = "%s/%s/%s/%s" % (config.Project.casia_dataset_b_path, id, dir, view)
				if not os.path.exists(img_dir):
					print("Error: %s do not exist" % img_dir)
	
def prepare_training_data(probe_dir, gallery_dir):
	view_list = ["%03d" % (18*i) for i in range(0, 11)]
	human_id_list = ["%03d" % i for i in range(1, 51)]
	human_id_list.remove('005') #000
	human_id_list.remove('026') #126
	human_id_list.remove('034')
	human_id_list.remove('037') #144
	human_id_list.remove('046')
	human_id_list.remove('048')

	probe_x = []
	probe_y = []
	gallery_x = []
	gallery_y = []
		
	paired_train_data = []

	# check dir exists
	check_dir(human_id_list, probe_dir, view_list)
	check_dir(human_id_list, gallery_dir, view_list)	

	# load data
	for id in human_id_list:
		logger.info("processing human %s" % id)
		for dir in probe_dir:
			for view in view_list:
				img_dir = "%s/%s/%s/%s" % (config.Project.casia_dataset_b_path, id, dir, view)
				probe_x.append(img_dir)
				probe_y.append(id)
		for dir in gallery_dir:
			for view in view_list:
				img_dir = "%s/%s/%s/%s" % (config.Project.casia_dataset_b_path, id, dir, view)
				gallery_x.append(img_dir)
				gallery_y.append(id)

	# pair data	
	x_range = len(human_id_list)-1
	view_range = len(view_list)-1
	view_coe = len(view_list)

	probe_coe = len(view_list)*len(probe_dir)
	probe_dir_range = len(probe_dir)-1

	gallery_coe = len(view_list)*len(gallery_dir)
	gallery_dir_range = len(gallery_dir)-1
	
	for i in range(2500):
		x = random.randint(0, x_range)
		i1 = random.randint(0, probe_dir_range)
		i2 = random.randint(0, gallery_dir_range)
		v = random.randint(0, view_range)

		idx1 = probe_coe*x+view_coe*i1+v
		idx2 = gallery_coe*x+view_coe*i2+v
		paired_train_data.append([probe_x[idx1], gallery_x[idx2], [0,1]])
		#paired_train_data.append([probe_x[idx1], gallery_x[idx2], [0,1] if probe_y[idx1]==gallery_y[idx2] else [1,0]])
		
		x1 = x
		x2 = random.randint(0, x_range)
		while(x2 == x1):
			x2 = random.randint(0, x_range)
		idx1 = probe_coe*x1+view_coe*i1+v
		idx2 = gallery_coe*x2+view_coe*i2+v
		paired_train_data.append([probe_x[idx1], gallery_x[idx2], [1,0]])
		#paired_train_data.append([probe_x[idx1], gallery_x[idx2], [0,1] if probe_y[idx1]==gallery_y[idx2] else [1,0]])

	return paired_train_data

def get_next_batch(img_class, probe_type, paired_train_data, batch_size=128):
	batch = sample(paired_train_data, batch_size)
	batch_x = []
	batch_y = []
	for item in batch:
		data1 = img_path_to_IMG(img_class, item[0])	
		data2 = img_path_to_IMG(img_class, item[1])	
		### data augmentation
		#rand = random.randint(1,10)
		#if rand%3==0:
		#	rand_angle1 = random.uniform(-8.0, 8.0)
		#	data1=misc.imrotate(data1, rand_angle1) 
		#	rand_angle2 = random.uniform(-8.0, 8.0)
		#	data2=misc.imrotate(data2, rand_angle2) 
		if len(data1.shape)>0 and len(data2.shape) > 0:
			batch_x.append([data1, data2])
			batch_y.append(item[2])
		else:
			print("GET_NEXT_BATCH: fail to extract %s or %s" % (item[0],item[1]))
	
	return np.asarray(batch_x), np.asarray(batch_y)
	
def load_data(img_class, data_class, probe_view, probe_dir, gallery_dir):
	probe_type=probe_dir[0][0:2].upper()
	if data_class == "validation":
		human_id_list = ["%03d" % i for i in range(51, 75)]
		human_id_list.remove('067')
		human_id_list.remove('068')
	elif data_class == "testing":
		human_id_list = ["%03d" % i for i in range(75, 125)]
		human_id_list.remove('079') #054
		human_id_list.remove('088') #054
		human_id_list.remove('109') #126
	else:
		print("Wrong data class")
		sys.exit(0)

	probe_x = []
	probe_y = []
	gallery_x = []
	gallery_y = []
	
	paired_x = []
	paired_y = []
	paired_data = []
	
	# check dir exists
	for id in human_id_list:
		for dir in probe_dir:
			img_dir = "%s/%s/%s/%s" % (config.Project.casia_dataset_b_path, id, dir, probe_view)
			if not os.path.exists(img_dir):
				logger.error("%s do not exist" % img_dir)
		for dir in gallery_dir:
			img_dir = "%s/%s/%s/%s" % (config.Project.casia_dataset_b_path, id, dir, probe_view)
			if not os.path.exists(img_dir):
				logger.error("%s do not exist" % img_dir)
	
	# get probe list
	for id in human_id_list:
		logger.info("processing human %s" % id)
		for dir in probe_dir:
				img_dir = "%s/%s/%s/%s" % (config.Project.casia_dataset_b_path, id, dir, probe_view)
				probe_x.append(img_dir)
				probe_y.append(id)
	
	view_list = ["000","018","036","054","072","090","108","126","144","162","180"]

	# get gallery list
	for id in human_id_list:
		for dir in gallery_dir:
			for view in view_list:
				img_dir = "%s/%s/%s/%s" % (config.Project.casia_dataset_b_path, id, dir, view)
				gallery_x.append(img_dir)
				gallery_y.append(id)

	x_range = len(human_id_list)-1
	view_coe = len(view_list)

	gallery_coe = len(view_list)*len(gallery_dir)
	gallery_dir_range = len(gallery_dir)-1

	# get probe data
	probe_imgs = [flatten(img_path_to_IMG(img_class, x)) for x in probe_x]
	angles = get_angle(img_class, probe_type,np.asarray(probe_imgs))
	#angles = np.asarray(len(probe_imgs)*[probe_view])

	# test accuracy
	#print(angles)	
	accuracy=sum(angles==[probe_view])*1.0/len(angles)
	print(accuracy)
	
	for probe_idx,angle in enumerate(angles):
		i = random.randint(0, gallery_dir_range)
		v = view_list.index(angle)
		gallery_idx = gallery_coe*(probe_idx//len(probe_dir))+view_coe*i+v
		gallery_img = flatten(img_path_to_IMG(img_class, gallery_x[gallery_idx]))
		probe_img = probe_imgs[probe_idx]		
		
		if len(probe_img) > 0 and len(gallery_img) > 0:
			paired_data.append([[np.asarray(probe_img), np.asarray(gallery_img)], [0,1]])
		else:
			print("LOAD_DATA: fail to extract %s of %s" % (img_dir, id))

		x = random.randint(0, x_range)
		gallery_idx = gallery_coe*(x)+view_coe*i+v
		while(gallery_y[gallery_idx]==probe_y[probe_idx]):
			x = random.randint(0, x_range)
			gallery_idx = gallery_coe*(x)+view_coe*i+v
		gallery_img = flatten(img_path_to_IMG(img_class, gallery_x[gallery_idx]))

		if len(probe_img) > 0 and len(gallery_img) > 0:
			paired_data.append([[np.asarray(probe_img), np.asarray(gallery_img)], [1,0]])
		else:
			print("LOAD_DATA: fail to extract %s of %s" % (img_dir, id))
	
	random.shuffle(paired_data)	

	paired_x = np.asarray([x.tolist() for x in np.asarray(paired_data)[:,0]])
	paired_y = np.asarray(paired_data)[:,1]
	
	return paired_x, paired_y

if __name__ == '__main__':
	level = logging.INFO
	FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
	logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

	train_view = ["000","018","036","054","072","090","108","126","144","162","180"]
	val_view = [["090"],["000","018"]]
	gallery_dir = ["nm-01","nm-02","nm-03","nm-04"]
	probe_dir = ["nm-06", "nm-05"]

	paired_train_data = prepare_training_data(train_view, probe_dir, gallery_dir)
	#for item in paired_train_data:
	#	print item
	get_next_batch(paired_train_data, batch_size=128)
	#load_validation_data()

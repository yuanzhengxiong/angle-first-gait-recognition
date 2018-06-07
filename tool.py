from skimage.io import imread
from skimage.io import imsave
from scipy.misc import imresize
import numpy as np
import tensorflow as tf
import os
import logging
import config

IMAGE_WIDTH = 88
IMAGE_HEIGHT = 128
NUM_PARTS=6
#PARTED_HEIGHT=[17, 19, 44, 44, 25, 23]
PARTED_HEIGHT=[16, 20, 44, 44, 24, 24]
PARTED_WIDTH=[88, 88, 44, 44, 88, 88]

def shift_left(img, left=10.0, is_grey=True):
    if 0 < abs(left) < 1:
        left = int(left * img.shape[1])
    else:
        left = int(left)

    img_shift_left = np.zeros(img.shape)
    if left >= 0:
        if is_grey:
            img_shift_left = img[:, left:]
        else:
            img_shift_left = img[:, left:, :]
    else:
        if is_grey:
            img_shift_left = img[:, :left]
        else:
            img_shift_left = img[:, :left, :]

    return img_shift_left


def shift_right(img, right=10.0):
    return shift_left(img, -right)


def shift_up(img, up=10.0, is_grey=True):
    if 0 < abs(up) < 1:
        up = int(up * img.shape[0])
    else:
        up = int(up)

    img_shift_up = np.zeros(img.shape)
    if up >= 0:
        if is_grey:
            img_shift_up = img[up:, :]
        else:
            img_shift_up = img[up:, :, :]
    else:
        if is_grey:
            img_shift_up = img[:up, :]
        else:
            img_shift_up = img[:up, :, :]

    return img_shift_up


def shift_down(img, down=10.0):
    return shift_up(img, -down)


def load_image_path_list(path):
    list_path = os.listdir(path)
    result = ["%s/%s" % (path, x) for x in list_path if x.endswith("jpg") or x.endswith("png")]
    return result


def image_path_list_to_image_data_list(image_path_list):
    image_data_list = []
    for image_path in image_path_list:
        im = imread(image_path)
        image_data_list.append(im)
    return image_data_list


def extract_human(img):
	left_blank = 0
	right_blank = 0
	
	up_blank = 0
	down_blank = 0
	
	height = img.shape[0]
	width = img.shape[1]
	
	for i in range(height):
		if np.sum(img[i, :]) == 0:
			up_blank += 1
		else:
			break
	
	for i in range(height-1, -1, -1):
		if np.sum(img[i, :]) == 0:
			down_blank += 1
		else:
			break
	
	for i in range(width):
		if np.sum(img[:, i]) == 0:
			left_blank += 1
		else:
			break
	
	for i in range(width-1, -1, -1):
		if np.sum(img[:, i]) == 0:
			right_blank += 1
		else:
			break
	
	img = shift_left(img, left_blank)
	img = shift_right(img, right_blank)
	img = shift_up(img, up_blank)
	img = shift_down(img, down_blank)
	return img

def center_person(img, size, method="simple"):
	best_index = 0
	origin_height, origin_width = img.shape
	if method == "simple":
		highest = 0
		for i in range(origin_width):
			data = img[:, i]
			for j, val in enumerate(data):
				# encounter body
				if val > 0:
					now_height = origin_height - j
					if now_height > highest:
						highest = now_height
						best_index = i
					break
	else:
		pixel_count = []
		for i in range(origin_width):
			pixel_count.append(np.count_nonzero(img[:, i]))
		count_all = sum(pixel_count)
		pixel_percent = [count * 1.0 / count_all for count in pixel_count]
		count_percent_sum = 0
		min_theta = 1
		for i, val in enumerate(pixel_percent):
			tmp = abs(0.5 - count_percent_sum)
			if tmp < min_theta:
				min_theta = tmp
				best_index = i
			count_percent_sum += val
	
	left_part_column_count = best_index
	right_part_column_count = origin_width - left_part_column_count - 1
	
	if left_part_column_count == right_part_column_count:
		return imresize(img, size)
	elif left_part_column_count > right_part_column_count:
		right_padding_column_count = left_part_column_count - right_part_column_count
		new_img = np.zeros((origin_height, origin_width + right_padding_column_count), dtype=np.int)
		new_img[:, :origin_width] = img
	else:
		left_padding_column_count = right_part_column_count - left_part_column_count
		new_img = np.zeros((origin_height, origin_width + left_padding_column_count), dtype=np.int)
		new_img[:, left_padding_column_count:] = img
	return imresize(new_img, size)


def build_GEI(img_list):
	norm_width = IMAGE_WIDTH 
	norm_height = IMAGE_HEIGHT
	result = np.zeros((norm_height, norm_width), dtype=np.int)
	
	human_extract_list = []
	for img in img_list:
		try:
			human_extract_list.append(center_person(extract_human(img), (norm_height, norm_width)))
		except:
			pass
		#	print("BUILD_GEI: fail to extract human from image")
	try:
		result = np.mean(human_extract_list, axis=0)
	except:
		print("BUILD_GEI: fail to calculate GEI, return an empty image")
	
	return result.astype(np.int32)

def img_path_to_GEI(img_path=''):
	id = img_path.replace("/", "_")
	cache_file = "%s/%s_GEI.npy" % (config.Project.test_data_path, id)
	if os.path.exists(cache_file) and os.path.isfile(cache_file):
		return np.load(cache_file)
	img_list = load_image_path_list(img_path)
	img_data_list = image_path_list_to_image_data_list(img_list)
	GEI_image = build_GEI(img_data_list)
	np.save(cache_file, GEI_image)
	return GEI_image

def segment_batch_img(batch_images):
	batch_part1=[]
	batch_part2=[]
	batch_part3=[]
	batch_part4=[]
	batch_part5=[]
	batch_part6=[]
	parted_batch_imgs=[]

	for IMG in batch_images:
		part1 = IMG[0:16,:]
		part2 = IMG[16:36,:]
		part3 = IMG[36:80,0:44]
		part4 = IMG[36:80,44:88]
		part5 = IMG[80:104,:]
		part6 = IMG[104:128,:]
		
		batch_part1.append(np.asarray(part1))
		batch_part2.append(np.asarray(part2))
		batch_part3.append(np.asarray(part3))
		batch_part4.append(np.asarray(part4))
		batch_part5.append(np.asarray(part5))
		batch_part6.append(np.asarray(part6))
	
	parted_batch_imgs=[np.asarray(batch_part1), np.asarray(batch_part2),
					   np.asarray(batch_part3), np.asarray(batch_part4),
		 			   np.asarray(batch_part5), np.asarray(batch_part6)]
	#imsave("%s/part1_.bmp" % config.Project.test_data_path, part1)
	#imsave("%s/part2_.bmp" % config.Project.test_data_path, part2)
	#imsave("%s/part3_.bmp" % config.Project.test_data_path, part3)
	#imsave("%s/part4_.bmp" % config.Project.test_data_path, part4)
	#imsave("%s/part5_.bmp" % config.Project.test_data_path, part5)
	#imsave("%s/part6_.bmp" % config.Project.test_data_path, part6)
	return parted_batch_imgs

def all_center(img_path):
	img_list = load_image_path_list(img_path)
	img_data_list = image_path_list_to_image_data_list(img_list)
	for idx, item in enumerate(img_data_list):
		print idx
		center_img=center_person(extract_human(item), (IMAGE_HEIGHT, IMAGE_WIDTH))
		imsave("./test/%s_center_img.bmp" % img_list[idx].replace("/","_").replace(".","_"),center_img)
	
	return

def GEI_to_GEnI(GEI):
	normalized_GEI = GEI/255.0
	GEnI=-1*normalized_GEI*np.log2(normalized_GEI)-(1-normalized_GEI)*np.log2(1-normalized_GEI)		
	GEnI[np.isnan(GEnI)] = 0.0
	return GEnI
	
def img_path_to_IMG(img_class, img_path):
	GEI = img_path_to_GEI(img_path)	
	if img_class == "GEI":
		return GEI
	elif img_class == "GEnI":
		GEnI = GEI_to_GEnI(GEI)
		return GEnI
	else:
		print("Error: Wrong img class")
		sys.exit(1)	

if __name__ == '__main__':
	import config
	img = imread(config.Project.casia_test_img, as_grey=True)
	
	extract_human_img = extract_human(img)
	human_extract_center = center_person(extract_human_img, (IMAGE_HEIGHT, IMAGE_WIDTH))
	
	#all_center(config.Project.casia_test_img_dir)
	#imsave("%s/origin_img.bmp" % config.Project.test_data_path, img)
	#imsave("%s/extract_human.bmp" % config.Project.test_data_path, extract_human_img)
	
	#imsave("%s/extract_human_center.bmp" % config.Project.test_data_path, human_extract_center)
	GEI_image = img_path_to_GEI(config.Project.casia_test_img_dir)
	GEnI = GEI_to_GEnI(GEI_image)
	#print GEnI
	imsave("%s/GEnI.bmp" % config.Project.test_data_path, GEnI) 
	#imsave("%s/GEI.bmp" % config.Project.test_data_path, GEI_image) 
	#rebuild_GEI(GEI_image)
	

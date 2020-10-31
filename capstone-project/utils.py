import torch
import torch.nn as nn
import torchvision
import os
import cv2
import numpy as np
import logging
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.dirname(__file__)) + '/'

def calculate_class_weights(data_dir, num_classes, indices):
	"""Calculate weights for each class"""
	
	trainId_to_count = dict(zip(range(num_classes), np.zeros(num_classes, np.uint8)))

	path = "dataset/train.txt"
	with open(path, 'r') as file:
		paths = file.readlines()

	mask_paths = []
	for path in paths:
		mask_path = (path.split('\t')[1]).split('\n')[0]
		mask_paths.append(mask_path)

	
	for step, label_img_path in enumerate(mask_paths):
	
		label_img = cv2.imread(f"{data_dir}/{label_img_path}", -1)

		for trainId in range(num_classes):
			# count how many pixels in label_img which are of object class trainId:
			trainId_count = np.sum(np.equal(label_img, trainId))
			# add to the total count:
			trainId_to_count[trainId] += trainId_count

	# compute the class weights according to the ENet paper:
	class_weights = []
	total_count = sum(trainId_to_count.values())
	for trainId, count in trainId_to_count.items():
		trainId_prob = float(count)/float(total_count)
		trainId_weight = 1/np.log(1.02 + trainId_prob)
		class_weights.append(trainId_weight)

	# calculate background class weight
	cls_counts = np.array(list(trainId_to_count.values()))
	bckgr_cls_count = total_count - sum(cls_counts[indices])
	weight = float(bckgr_cls_count)/float(total_count)
	weight =  1/np.log(1.02 + weight)

	class_weights = (np.array(class_weights)[indices]).tolist()
	return class_weights + [weight]

def snapshot(model, loss, out_dir ,suffix = 1):
	"""This fun saves the model and optimizer state dictionaries.
	Keep in mind to change the 'map_location' argument in torch.load while loading the model on cpu."""
	
	path = f'{out_dir}/'
	
	checkpoint = {"model": model,
				 "state_dict": model.state_dict(),
				 'loss':loss}
	
	torch.save(checkpoint, path + ('epoch_' + str(suffix) +'.pth'))
	
	
def load_model(epoch_number, dir_type):
	path = 'checkpoints/{}/'.format(dir_type)
	checkpoint = torch.load( path + 'epoch_' + str(epoch_number) + '.pth')
	model = checkpoint['model']
	model.load_state_dict(checkpoint['state_dict'])
	optimizer = checkpoint['optimizer']
	
	return model, optimizer

def save_mask(im, filename, out_dir):
	'''Save as image from pytorch tensor'''
	path = out_dir
	
	im = Image.fromarray(im)
	im.save(path + filename)
	
def save_image(im, filename, out_dir):
	path = out_dir

	mean, std = [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
	normalized_im = im.detach().cpu().numpy()
	normalized_im = np.transpose(normalized_im, (1, 2, 0))
	normalized_im = (std * normalized_im) + mean
	normalized_im = normalized_im.astype(np.uint8)
	normalized_im = np.clip(normalized_im, 0, 255)
	normalized_im = Image.fromarray(normalized_im)
	normalized_im.save(path + filename)


def get_logger(logdir):
	logger = logging.getLogger("main")
	ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
	ts = ts.replace(":", "_").replace("-", "_")
	file_path = ROOT + logdir + "/run_{}.log".format(ts)
	hdlr = logging.FileHandler(file_path)
	formatter = logging.Formatter("%(asctime)s %(message)s")
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)
	logger.setLevel(logging.INFO)
	return logger

def save_plot(data, path):

	path = path + '/plots/'

	if not os.path.exists(path):
		os.makedirs(path)

	if isinstance(data, dict):
		keys = list(data.keys())
		fig = plt.figure()
		plt.plot(data[keys[0]])
		plt.plot(data[keys[1]])
		plt.legend((keys[0], keys[1]), loc = 'upper right')
		plt.grid()
		plt.title('Losses')
		plt.xlabel('epochs')
		plt.ylabel('loss')
		plt.savefig(path + 'loss.png')
		plt.close()

	elif isinstance(data, list):

		fig = plt.figure()
		plt.plot(data)
		plt.grid()
		plt.title('Metric')
		plt.xlabel('epochs')
		plt.ylabel('mIoU')
		plt.savefig(path + 'metric.png')
		plt.close()
	else:
		raise ValueError('Data should be either list or dictionary')
	

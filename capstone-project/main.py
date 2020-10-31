import argparse
import numpy as np
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
import time
import os
import logging

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

from loader import *
from models.unet import *
from utils import *
from lr_scheduler import LR_Scheduler
from loss import SegmentationLoss
from metric import *


# class names in images which have to be segmented
names = [v for (k,v) in class_names.items() if mapping.get(k) != ignore_label] + ['background']
CLASSES = CustomDataset.num_classes

def get_loaders(dir, bs, size= 512):
	"""Create dataloaders for training"""

	train_data = CustomDataset(dir, mode = 'train', crop_size=size)
	validation_data = CustomDataset(dir, mode = 'val', crop_size=size)
	
	# creating dataloaders for train, val
	train_loader = DataLoader(train_data, batch_size= bs, num_workers=5, shuffle= True, pin_memory=True)
	val_loader = DataLoader(validation_data, batch_size= bs, num_workers=5, shuffle= True, pin_memory=True)

	loaders = {'train': train_loader, 'val':val_loader}
	
	return loaders

def initialize_network(name):
	
	"""Initialize the network"""
	
	if name == 'unet':
		network = Unet(CLASSES)
	elif name == 'pspnet':
		pass
		
	return network

def get_optimizer(lr, network, train_loader, epochs):
	"""Optimizer is SGD with momentum and poly scheduler"""
	
	optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay= 1e-4)
	scheduler = LR_Scheduler('poly', lr, epochs, len(train_loader), lr_step=1)
	return optimizer, scheduler


def define_loss(loss_type, data_dir, device, aux_weight = None , use_weights = True):
	"""Define loss"""

	class_weights = None
	if use_weights:
		indices = [k for (k,v) in mapping.items() if v != ignore_label]
		class_weights = calculate_class_weights(data_dir, 34, indices)
		
		# adding background weight as 0, it does not matter because it will be ignored by loss
		# class_weights = class_weights
	
	criterion = SegmentationLoss(loss_type, class_weights, aux_weight)

	return criterion.to(device)


def train(network, criterion, optimizer, scheduler, loaders, epochs, args):

	val_iou, losses = [], {'train_loss': [], 'val_loss': []}

	val_metrics = runningScore(CLASSES)
	val_loss_meter = averageMeter()
	train_loss_meter = averageMeter()
	logger = get_logger(args.exp_name)
	logger.info("Loss {}".format(args.loss_type))

	for epoch in range(epochs):

		network = network.train()
		curr_lr = optimizer.param_groups[0]['lr']
		print(f"Epoch {epoch}, Learning rate: {curr_lr} ")
		logger.info(f"Epoch {epoch}, Learning rate: {curr_lr} ")

		for idx, (im, mask ) in tqdm(enumerate(loaders['train']), ascii = True, desc = "Training"):
			scheduler(optimizer, idx, epoch)
			optimizer.zero_grad()

			im = im.to(device)
			mask = mask.to(device)
			
			outputs = network(im)
			loss = criterion(outputs, mask)
			loss.backward()
			optimizer.step()
			
			train_loss_meter.update(loss.item())
			
		# calculate total loss
		print_str = f'Training loss : %.4f ' % (train_loss_meter.avg)
		losses['train_loss'].append(train_loss_meter.avg)
		print(print_str)

		# log the loss
		logger.info(print_str)
		
		# on each epoch check the results on validation set
		if epoch%1 == 0:
			network.eval()
			flag = True
			with torch.no_grad():
				
				# choose a random image to save from validation set
				rand = random.randint(0,50)
				for idx, (im, labels_val) in tqdm(enumerate(loaders['val']), ascii = True, desc = "Validation"):

					im = im.to(device)
					labels_val = labels_val.to(device)

					outputs = network(im)
					val_loss = criterion(outputs, labels_val)

					pred = outputs.data.max(1)[1].cpu().numpy()
					gt = labels_val.data.cpu().numpy()

					val_metrics.update(gt, pred)
					val_loss_meter.update(val_loss.item())
					
					if epoch%10 == 0 and flag:
						
						snapshot(network, criterion, args.exp_name, epoch)
						# save original im, actual mask and generated mask 
						save_image(im[0], f'im_epoch_{epoch}.png', out_dir)
						save_mask(colorize_mask(labels_val[0]), f'mask_epoch_{epoch}.png', out_dir)
						save_mask(colorize_mask(pred[0]), f'pred_epoch_{epoch}.png', out_dir)

						flag = False
					
			
			score, class_iou = val_metrics.get_scores()
			class_iou = dict(zip(names, class_iou.values()))
			print_str = f"Val loss: %.4f, " % (val_loss_meter.avg) + f"score: {score}, class_IoU: {class_iou}"
			print(print_str)
			logger.info(print_str)
			val_iou.append(score['Mean IoU'])
			losses['val_loss'].append(val_loss_meter.avg)

			val_loss_meter.reset()
			val_metrics.reset()

	return val_iou, losses
			
	

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description = "Octave Unet experiments")
	
	parser.add_argument('-bs', '--batch_size', help = 'Batch size', default= 12, type=int )
	parser.add_argument('-size', '--img_size', help = 'Image size', default= 704, type=int )
	parser.add_argument('-lr', '--learning_rate', help = 'Learning rate', default= 0.01, type=float )
	parser.add_argument('--data_dir', help = 'Path for data directory', default = '/ds/images/Cityscapes/')
	parser.add_argument('--epochs', help = 'Number of epochs', default= 200, type=int )
	parser.add_argument('--device', help = 'Device type', default = 'cuda:0')
	parser.add_argument('--loss_type', help = 'Loss type', default = 'cross_entropy')
	parser.add_argument('-index', '--ignore_index', help = 'Index to ignore while calculating loss', default= None)
	parser.add_argument('--net_name', help = 'Name of network', default= 'unet')
	parser.add_argument('--exp_name', help = 'Name of the experiment', default= 'exp_unet3')

	args = parser.parse_args()
	lr = args.learning_rate
	epochs = args.epochs
	bs = args.batch_size
	loss = args.loss_type
	network_name = args.net_name
	img_size = args.img_size
	data_dir = args.data_dir
	exp_name = args.exp_name

	if not os.path.exists(exp_name):
		os.makedirs(exp_name+'/')

	out_dir = (exp_name + '/output/images/')
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	start = datetime.datetime.now()
	
	device = args.device
	network = initialize_network(network_name).to(device)
	loaders = get_loaders(data_dir, bs, size = img_size)
	optimizer, scheduler = get_optimizer(lr, network, loaders['train'], epochs)
	
	# start training
	criterion = define_loss(loss, data_dir, device)
	val_iou, losses = train(network, criterion, optimizer, scheduler, loaders, epochs, args)
		
	end = datetime.datetime.now()
	save_plot(val_iou, os.path.dirname(out_dir[:-1]))
	save_plot(losses, os.path.dirname(out_dir[:-1]))
	
	print("\n Training finished, total time taken: ", end - start)
	


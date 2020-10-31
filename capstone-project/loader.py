import torch
import os 
import random
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image, ImageOps


ROOT = os.path.dirname(os.path.abspath(__file__))

# ignore this label while calculating loss
ignore_label = 19

# class names present in images
class_names = {0:'unlabeled' , 1:'ego vehicle', 2: 'rect border', 3:  'out of roi', 4: 'static',
			5: 'dynamic', 6: 'ground', 7: 'road', 8: 'sidewalk', 9: 'parking', 10: 'rail track',
			11: 'building', 12: 'wall', 13: 'fence', 14: 'guard rail', 15: 'bridge', 16: 'tunnel',
			17: 'pole', 18: 'polegroup', 19: 'traffic light', 20: 'traffic sign', 21: 'vegetation',
			22: 'terrain', 23: 'sky', 24: 'person', 25: 'rider', 26: 'car', 27: 'truck', 28: 'bus',
			29: 'caravan', 30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle', -1: 'licenseplate'}


mapping = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
						3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
						7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
						14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
						18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
						28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

# color palette
palette = [
		[128, 64, 128],
		[244, 35, 232],
		[70, 70, 70],
		[102, 102, 156],
		[190, 153, 153],
		[153, 153, 153],
		[250, 170, 30],
		[220, 220, 0],
		[107, 142, 35],
		[152, 251, 152],
		[0, 130, 180],
		[220, 20, 60],
		[255, 0, 0],
		[0, 0, 142],
		[0, 0, 70],
		[0, 60, 100],
		[0, 80, 100],
		[0, 0, 230],
		[119, 11, 32],
		[0,0,0]
	]

# color for each class
label_color = dict(zip(range(20), palette))


def colorize_mask(mask):
	"""Colorize the output mask from the network"""
	
	if torch.is_tensor(mask):
		new_mask = mask.detach().cpu().numpy()
	else:
		new_mask = mask
	if len(new_mask.shape) == 3:
		new_mask = np.argmax(new_mask, 0)

	r = np.zeros(new_mask.shape, dtype = np.uint8)
	g = np.zeros(new_mask.shape, dtype = np.uint8)
	b = np.zeros(new_mask.shape,  dtype = np.uint8)
	for i in range(20):
		r[new_mask == i] = label_color[i][0]
		g[new_mask == i] = label_color[i][1]
		b[new_mask == i] = label_color[i][2]
		
	mask = np.stack((r,g,b), 0)
	mask = np.transpose(mask, (1,2,0))
	return mask


class CustomDataset(Dataset):
	'''This class is meant to read the images from all the directories and return the whole dataset for 
	the purpose of Dataloader in pytorch.'
	
	Parameters:

		mode: train or test or validate
		new_H: height fot the resized image
		new_W: width for the resized image
		transforms: boolean value to apply transforms
	'''

	num_classes = 20

	def __init__(self, dir, mode = 'train', crop_size = 768 ,scale = True):
		
		self.data_dir = dir
		self.mode = mode
		self.images = list()
		self.target_im = list()
		self.mean, self.std = [.485, .456, .406], [.229, .224, .225]
		self.crop_size = crop_size
		self.scale = scale
		
		if mode == 'train':
			f = open('dataset/train.txt', 'r')
			paths = f.readlines()
		elif mode == 'val':
			f = open('dataset/val.txt', 'r')
			paths = f.readlines()
		else:
			f = open('dataset/test.txt', 'r')
			paths = f.readlines()

		# print(os.listdir(self.data_dir))

		# read all the images and mask from directory 
		for filename in paths:
			filename = filename.split('\t')
			img, label = filename[0], filename[1].replace('\n','')
			self.images.append(os.path.join(self.data_dir , img))
			self.target_im.append(os.path.join(self.data_dir , label))

		assert len(self.images) == len(self.target_im), 'Number of Images are not same as masks available'
				
	def __repr__(self):
		'''String return when class object is called '''

		string = 'Class {}\n'.format(self.__class__.__name__)
		string += ' Number of datapoints: {}\n'.format(len(self.images))
		string += ' Root location {}\n'.format(self.dir)
		string += ' Split ' + self.mode 
		string += ' \nNumber of classes ' + str(self.classes)
		return string
	
	
	def ids_to_class(self, mask):
		'''
		Maps pixel values of a mask to class ids.
		Make sure these class ids are mapped properly in the dict mapping 

		Parameters:
			mask(numpy array): Ground truth mask for an image

		Returns:
			target_mask(numpy array): class ids mapped mask
		'''

		target_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype = np.uint8)
		for k,v in mapping.items():
			target_mask[mask == k] = v

		return target_mask


	def __getitem__(self, index):
		'''Prepare image and mask based on the index'''

		img = Image.open(self.images[index]).convert('RGB')
		mask = Image.open(self.target_im[index]).convert('L')
		
		if self.mode == 'train':
			img, mask = self._sync_transform(img, mask)
		else:
			img, mask = self._val_sync_transform(img, mask)
		
		img = self._transform(img)
		
		mask = np.array(mask, dtype = np.uint8)

		# prepare masks for training purpose
		mask = self.ids_to_class(mask)
		target_mask = torch.from_numpy(mask).long()

		
		return img, target_mask
	
	def __len__(self):
		'''Calculates length of the whole dataset in current mode.'''

		return len(self.images) 


	def _transform(self, image):
		'''Convert image to tensor and normalize it.'''
		
		tf_image = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=self.mean, std=self.std)
		])

		return tf_image(image)
	
	def _sync_transform(self,img, mask):
		"""
		Apply random horizontal flipping, scaling to image and mask. 
		Used as training augmentation
		"""
		
		# random mirror
		if random.random() < 0.5:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
			mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

		crop_size = self.crop_size
			
		if self.scale:
			base_size = 900
			short_size = random.randint(int(base_size*0.90), int(base_size*1.3))
		else:
			short_size = crop_size

		w, h = img.size
		if h > w:
			ow = short_size
			oh = int(1.0 * h * ow / w)
		else:
			oh = short_size
			ow = int(1.0 * w * oh / h)
			
		img = img.resize((ow, oh), Image.BILINEAR)
		mask = mask.resize((ow, oh), Image.NEAREST)

		# random crop crop_size
		w, h = img.size
		x1 = random.randint(0, w - crop_size)
		y1 = random.randint(0, h - crop_size)
		img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
		mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
		return img, mask
	
	def _val_sync_transform(self, img, mask):
		'''Crop image of given size without any othe data augmentation'''
		
		outsize = self.crop_size
		short_size = outsize
		w, h = img.size
		if w > h:
			oh = short_size
			ow = int(1.0 * w * oh / h)
		else:
			ow = short_size
			oh = int(1.0 * h * ow / w)
		img = img.resize((ow, oh), Image.BILINEAR)
		mask = mask.resize((ow, oh), Image.NEAREST)

		# center crop
		w, h = img.size
		x1 = int(round((w - outsize) / 2.))
		y1 = int(round((h - outsize) / 2.))
		img = img.crop((x1, y1, x1+outsize, y1+outsize))
		mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
		# final transform
		return img, mask



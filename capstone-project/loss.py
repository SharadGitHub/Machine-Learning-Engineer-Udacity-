import torch
import torch.nn as nn


class SegmentationLoss(nn.Module):
	"""
	Calculates loss

	Supports three segmentation losses ['cross_entropy', 'dice_loss', 'cross_entropy_with_dice']

	Attributes:
		loss_name (str): type of loss
		class_weight (tensor): weights for classed in case of imbalance class distribution
		aux_weight (float): weight for dice loss if cross_entropy_with_dice is used
		ignore_index (int): class id which will be ignored while calculating loss
	"""

	def __init__(self, loss_name = 'dice_loss', class_weight = None, aux_weight = 1.0, ignore_index = -1):
		super(SegmentationLoss, self).__init__()
		self.loss_name = loss_name
		self.aux_weight = aux_weight
		self.ignore_index = ignore_index
		self.class_weight = class_weight

		if class_weight != None and not torch.is_tensor(class_weight):
			self.class_weight = torch.tensor(class_weight).type(torch.float)
			
		
		self.loss_fn = self._loss_func()
	
	def _loss_func(self):
		'''A factory method to initialize the right loss'''

		if self.loss_name == 'dice_loss':

			return self.dice_loss

		elif self.loss_name == 'cross_entropy':
			return nn.CrossEntropyLoss(self.class_weight, ignore_index = self.ignore_index)

		elif self.loss_name == 'cross_entropy_with_dice':
			return self.entropy_with_dice
		else:
			NotImplementedError


	def forward(self, prediction, target):
		return self.loss_fn(prediction, target)


	def dice_loss(self, x, target, smooth = 1.0):
		'''Implementation of dice loss'''

		if self.class_weight is None:
			self.class_weight = torch.ones(x.shape[1])

		x = x.clone().cpu()
		target = target.clone().cpu()

		x = nn.functional.softmax(x, dim = 1)

		if target.shape != x.shape:
			encoded_target = torch.zeros(x.shape).scatter_(1, target.unsqueeze(1),1)
		else:
			encoded_target = target


		intersection = x * encoded_target
		denominator = (x * x) + (encoded_target * encoded_target)

		denominator = denominator.sum(3).sum(2)
		numerator = intersection.sum(3).sum(2)
		
		l = (2 * numerator + smooth) / (denominator + smooth)

		if self.ignore_index != -1:
			self.class_weight = torch.cat((self.class_weight[:self.ignore_index], self.class_weight[self.ignore_index+1:]))
			l = torch.cat((l[:,0:self.ignore_index], l[:,self.ignore_index+1:]), dim = 1)

		# take weighted average for classes
		l = (l * self.class_weight)
		return 1 - (l.sum(1)/ torch.sum(self.class_weight)).mean()


	def entropy_with_dice(self, x, target):
		'''Both cross entropy and weighted dice loss'''

		if self.aux_weight is None:
			raise ValueError
		
		return  nn.functional.cross_entropy(x, target, self.class_weight, ignore_index = self.ignore_index) +  \
				(self.aux_weight * self.dice_loss(x, target))
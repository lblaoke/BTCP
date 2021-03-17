import torch
from torch import nn
import torchvision
from models.BaseModels import *

#VGG16 network (can be resumed from pretained models)
class VGG16(nn.Module):
	def __init__(self,input_shape,num_feature,num_class,transfered=None):
		super(VGG16,self).__init__()

		#load transfered encoder
		vgg16 = torchvision.models.vgg16_bn(pretrained=False)
		if transfered:
			vgg16.load_state_dict(torch.load(transfered))

		self.conv = vgg16.features

		self.fc1 = nn.Sequential( #512*(input_shape[1]/32)*(input_shape[2]/32)
			nn.Linear(512*(input_shape[1]//32)*(input_shape[2]//32),num_feature)	,
			nn.ReLU()								,
			nn.Dropout(0.5)
		) #num_feature

		self.fc2 = nn.Linear(num_feature,num_class)

		self.confidnet = nn.Sequential( #num_feature
			nn.Linear(num_feature,400)	,
			nn.ReLU()			,
			nn.Linear(400,400)		,
			nn.ReLU()			,
			nn.Linear(400,400)		,
			nn.ReLU()			,
			nn.Linear(400,400)		,
			nn.ReLU()			,
			nn.Linear(400,1)		,
			nn.Hardsigmoid()
		) #1

	def forward(self,x,mode='all'):
		x = self.conv(x)
		x = x.view(x.size(0),-1)
		x = self.fc1(x)

		#for classifier only
		if mode=='fc':
			y = self.fc2(x)
			y_min,_ = y.min(dim=1,keepdim=True)
			y = y-y_min
			y = y/y.sum(dim=1,keepdim=True)
			return y

		#for confidence prediction only
		if mode=='confidnet':
			return self.confidnet(x)

		#get both results at one time
		y,conf = self.fc2(x),self.confidnet(x)
		y_min,_ = y.min(dim=1,keepdim=True)
		y = y-y_min
		y = y/y.sum(dim=1,keepdim=True)
		return y,conf

from torch import nn
from models.BaseModels import *

#small convolutional network for MNIST dataset
class ConvNetMNIST(nn.Module):
	def __init__(self,input_shape,num_feature,num_class,transfered=None):
		super(ConvNetMNIST,self).__init__()

		self.conv = nn.Sequential( #input_shape[0]*input_shape[1]*input_shape[2]
			Conv2dSame(input_shape[0],32,3)	,
			nn.ReLU()			,
			Conv2dSame(32,64,3)		,
			nn.ReLU()			,
			nn.MaxPool2d(2)			,
			nn.Dropout(0.25)
		) #64*(input_shape[1]/2)*(input_shape[2]/2)

		self.fc1 = nn.Sequential( #64*(input_shape[1]/2)*(input_shape[2]/2)
			nn.Linear(64*(input_shape[1]//2)*(input_shape[2]//2),num_feature)	,
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

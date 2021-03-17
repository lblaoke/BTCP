from torch import nn
from models.BaseModels import *

#small convolutional network for SVHN dataset (format 2)
class ConvNetSVHN2(nn.Module):
	def __init__(self,input_shape,num_feature,num_class,transfered=None):
		super(ConvNetSVHN2,self).__init__()

		self.conv = nn.Sequential( #input_shape[0]*input_shape[1]*input_shape[2]
			Conv2dSame_BN_ReLU(input_shape[0],32,3)	,
			Conv2dSame_BN_ReLU(32,32,3)		,
			nn.MaxPool2d(2)				,
			nn.Dropout(0.3)				,
			Conv2dSame_BN_ReLU(32,64,3)		,
			Conv2dSame_BN_ReLU(64,64,3)		,
			nn.MaxPool2d(2)				,
			nn.Dropout(0.3)				,
			Conv2dSame_BN_ReLU(64,128,3)		,
			Conv2dSame_BN_ReLU(128,128,3)		,
			nn.MaxPool2d(2)				,
			nn.Dropout(0.3)
		) #128*(input_shape[1]/8)*(input_shape[2]/8)

		self.fc1 = nn.Sequential( #128*(input_shape[1]/8)*(input_shape[2]/8)
			nn.Linear(128*(input_shape[1]//8)*(input_shape[2]//8),num_feature)	,
			nn.ReLU()								,
			nn.Dropout(0.3)
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

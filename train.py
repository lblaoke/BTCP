import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from itertools import cycle
import argparse
import yaml
from importlib import import_module
from helpers.Devices import *
from helpers import MyDataset
from helpers.Tricks import *

#extract configuration
parser = argparse.ArgumentParser()
parser.add_argument('--gamma',type=float,default=0.)
parser.add_argument('--conf',type=str)
parser.add_argument('--device',type=str)
parser.add_argument('--time',type=str,default='.')
arg = parser.parse_args()

with open(arg.conf) as f:
	conf = yaml.load(f,Loader=yaml.FullLoader)

#use a free device
device_id = free_device_id(arg.device)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
print('on device %d' % device_id)

#load model
package = import_module('models.'+conf['model']['name'])
net_class = getattr(package,conf['model']['name'])
net = net_class(
	input_shape	= conf['dataset']['shape']	,
	num_feature	= conf['model']['num_feature']	,
	num_class	= conf['dataset']['num_class']	,
)
net.load_state_dict(torch.load('.cache/pretrained/%s/classifier_%s.pkl' %(arg.time,conf['dataset']['name'])))
net = net.to(device_id)

#set hyperparameters
opt_class = getattr(torch.optim,conf['training']['optimizer']['name'])
loss_func1 = getattr(torch.nn,conf['training']['loss_classifier'])()
loss_func2 = getattr(torch.nn,conf['training']['loss_confidnet'])(reduction='none')
loss_func1 = loss_func1.to(device_id)
loss_func2 = loss_func2.to(device_id)

#load dataset
dataSet_class = getattr(MyDataset,conf['dataset']['name'])
dataSet = dataSet_class(conf['dataset']['path'],'train')

if __name__=='__main__':

	#train confidnet
	set_trainable(net.conv,False)
	set_trainable(net.fc1,False)
	opt = opt_class(
		params		= net.parameters()				,
		lr		= conf['training']['optimizer']['lr']		,
		momentum	= conf['training']['optimizer']['momentum']	,
		weight_decay	= conf['training']['optimizer']['weight_decay']
	)
	dataLoader = DataLoader(
		dataset		= dataSet				,
		batch_size	= conf['training']['batch_size']	,
		shuffle		= True					,
		num_workers	= conf['training']['num_worker']	,
		drop_last	= conf['training']['drop_last']
	)
	net = net.train()
	for i in tqdm(range(conf['training']['confidnet_epoch']),ncols=70):
		for _,(X,y) in enumerate(dataLoader):
			X,y = X.to(device_id),y.to(device_id)

			#forward calculation
			opt.zero_grad()
			y_hat,c_hat = net(X)

			#generate supervision signals for confidence prediction
			c = y_hat.detach().gather(1,y.unsqueeze(1))
			confc = c.flatten()

			#generate masks for loss constraints
			density_hist = confc.histc(bins=10,min=0.,max=1.)/len(confc)
			index = (confc*10.).long()
			index[index==10] = 9
			sensitivity = (1.-density_hist[index])**arg.gamma

			#back propagation
			loss_confidnet = F.linear(loss_func2(c_hat,c).flatten(),sensitivity)/len(sensitivity)
			loss_confidnet.backward()
			opt.step()

	#fine-tuning
	set_trainable(net.conv,True)
	set_trainable(net.fc1,True)
	opt = opt_class(
		params		= net.parameters()				,
		lr		= conf['training']['optimizer']['lr_finetuning'],
		momentum	= conf['training']['optimizer']['momentum']	,
		weight_decay	= conf['training']['optimizer']['weight_decay']
	)
	for _ in tqdm(range(conf['training']['finetuning_epoch']),ncols=70):
		for _,(X,y) in enumerate(dataLoader):
			X,y = X.to(device_id),y.to(device_id)

			#forward calculation
			opt.zero_grad()
			y_hat,c_hat = net(X)

			#generate supervision signals for confidence prediction
			c = y_hat.detach().gather(1,y.unsqueeze(1))
			confc = c.flatten()

			#generate masks for loss constraints
			density_hist = confc.histc(bins=10,min=0.,max=1.)/len(confc)
			index = (confc*10.).long()
			index[index==10] = 9
			sensitivity = (1.-density_hist[index])**arg.gamma

			#back propagation
			loss_classifier = loss_func1(y_hat,y)
			loss_confidnet = F.linear(loss_func2(c_hat,c).flatten(),sensitivity)/len(sensitivity)
			loss = (loss_classifier+loss_confidnet)/2.
			loss.backward()
			opt.step()

	#save parameters
	torch.save(net.state_dict(),'.cache/trained/%s/confidnet_%s_gamma=%f.pkl' %(arg.time,conf['dataset']['name'],arg.gamma))

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
from itertools import cycle
import argparse
import yaml
from importlib import import_module
from helpers.Devices import *
from helpers import MyDataset
from helpers.Tricks import *

#extract configuration
parser = argparse.ArgumentParser()
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

#initialize model
package = import_module('models.'+conf['model']['name'])
net_class = getattr(package,conf['model']['name'])
net = net_class(
	input_shape	= conf['dataset']['shape']	,
	num_feature	= conf['model']['num_feature']	,
	num_class	= conf['dataset']['num_class']	,
	transfered	= conf['model']['transfered']
)
net = net.to(device_id)

#set hyperparameters
opt_class = getattr(torch.optim,conf['training']['optimizer']['name'])
loss_func1 = getattr(torch.nn,conf['training']['loss_classifier'])()
loss_func1 = loss_func1.to(device_id)

#load dataset
dataSet_class = getattr(MyDataset,conf['dataset']['name'])
dataSet = dataSet_class(conf['dataset']['path'],'train')

if __name__=='__main__':

	#train classifier
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
	if conf['model']['transfered']:
		print('transfered from '+conf['model']['transfered'])
		set_trainable(net.conv,False)
	net = net.train()
	for _ in tqdm(range(conf['training']['classifier_epoch']),ncols=70):
		for _,(X,y) in enumerate(dataLoader):
			X,y = X.to(device_id),y.to(device_id)	

			#calcualte estimated results
			opt.zero_grad()
			y_hat = net(X,'fc')

			#calculate loss and propagate back
			loss_classifier = loss_func1(y_hat,y)
			loss_classifier.backward()
			opt.step()

	#retrain transfered parts (if needed)
	if conf['model']['transfered']:
		opt = opt_class(
			params		= net.parameters()				,
			lr		= conf['training']['optimizer']['lr_finetuning'],
			momentum	= conf['training']['optimizer']['momentum']	,
			weight_decay	= conf['training']['optimizer']['weight_decay']
		)
		set_trainable(net.conv,True)
		for _ in tqdm(range(conf['training']['finetuning_epoch']),ncols=70):
			for _,(X,y) in enumerate(dataLoader):
				X,y = X.to(device_id),y.to(device_id)

				#calcualte estimated results
				opt.zero_grad()
				y_hat = net(X,'fc')

				#calculate loss and propagate back
				loss_classifier = loss_func1(y_hat,y)
				loss_classifier.backward()
				opt.step()

	#save parameters and labels
	torch.save(net.state_dict(),'.cache/pretrained/%s/classifier_%s.pkl' %(arg.time,conf['dataset']['name']))

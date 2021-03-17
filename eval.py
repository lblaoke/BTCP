import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import yaml
from importlib import import_module
from helpers.Devices import *
from helpers import Metrics
from helpers import MyDataset

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

#load model
package = import_module('models.'+conf['model']['name'])
net_class = getattr(package,conf['model']['name'])
net = net_class(
	input_shape	= conf['dataset']['shape']	,
	num_feature	= conf['model']['num_feature']	,
	num_class	= conf['dataset']['num_class']
)
net.load_state_dict(torch.load('.cache/trained/%s/confidnet_%s_gamma=%f.pkl' %(arg.time,conf['dataset']['name'],arg.gamma)))
net = net.to(device_id)

#load dataset
dataSet_class = getattr(MyDataset,conf['dataset']['name'])
dataSet = dataSet_class(conf['dataset']['path'],'t10k')

dataLoader = DataLoader(
	dataset		= dataSet				,
	batch_size	= conf['evaluation']['batch_size']	,
	shuffle		= False					,
	num_workers	= conf['evaluation']['num_worker']	,
	drop_last	= conf['evaluation']['drop_last']
)

#set up result containers
y_pred = torch.empty(0,1,dtype=torch.float32)
y_true = dataSet.y.unsqueeze(dim=1)
conf_pred = torch.empty(0,1,dtype=torch.float32)

if __name__=='__main__':
	net = net.eval()
	for _,(X,_) in enumerate(dataLoader):
		X = X.to(device_id)

		#predict and generate confidence labels
		with torch.no_grad():
			y_hat,c_hat = net(X)
		y_hat = y_hat.argmax(axis=1).unsqueeze(dim=1)
		y_hat,c_hat = y_hat.cpu(),c_hat.cpu()

		#collect results
		y_pred = torch.cat([y_pred,y_hat],dim=0)
		conf_pred = torch.cat([conf_pred,c_hat],dim=0)

	#evaluate classifier
	conf_true = (y_pred==y_true)
	print('classifier accuracy =',conf_true.sum().item()/len(conf_true))

	#evaluate confidnet
	for metric in conf['evaluation']['metrics']:
		metric_func = getattr(Metrics,metric)
		print('confidnet %s =' % metric,metric_func(conf_pred,conf_true))

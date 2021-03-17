from os.path import join
from struct import unpack
import numpy as np
import torch
import pickle
from scipy.io import loadmat
from torch.utils.data import Dataset

class MnistDataset(Dataset):
	def __init__(self,path,kind):
		assert kind=='train' or kind=='t10k','Unsupported kind '+kind

		#generate full path
		labels_path = join(path,'%s-labels.idx1-ubyte' % kind)
		images_path = join(path,'%s-images.idx3-ubyte' % kind)

		#open files and read
		with open(labels_path,'rb') as lbpath:
			magic,n = unpack('>II',lbpath.read(8))
			y = np.fromfile(lbpath,dtype=np.uint8)
		with open(images_path,'rb') as imgpath:
			magic,num,rows,cols = unpack('>IIII',imgpath.read(16))
			X = np.fromfile(imgpath,dtype=np.uint8).reshape(len(y),1,28,28)

		#check size of dataset
		if kind=='train':
			assert X.shape==(60000,1,28,28),'Data missed partially, expect (60000,1,28,28), but got '+str(X.shape)+' instead'
			assert y.shape==(60000,),'Data missed partially, expect (60000,), but got '+str(y.shape)+' instead'
		else:
			assert X.shape==(10000,1,28,28),'Data missed partially, expect (10000,1,28,28), but got '+str(X.shape)+' instead'
			assert y.shape==(10000,),'Data missed partially, expect (10000,), but got '+str(y.shape)+' instead'

		#convert all to tensors
		self.X = torch.from_numpy(X.astype(np.float32))
		self.y = torch.from_numpy(y.astype(np.int64))

	def __getitem__(self,index):
		return self.X[index],self.y[index]

	def __len__(self):
		return self.y.size(0)

class FashionMnistDataset(MnistDataset):
	def __init__(self,path,kind):
		super(FashionMnistDataset,self).__init__(path,kind)

class Cifar10Dataset(Dataset):
	def __init__(self,path,kind):
		X = np.empty(shape=(0,3,32,32),dtype=np.uint8)
		y = np.empty(shape=(0,),dtype=np.int32)

		#open files and read
		if kind=='train':
			for batch in range(1,6):
				p = join(path,'data_batch_%d' % batch)
				with open(p,'rb') as f:
					d = pickle.load(f,encoding='bytes')
				X = np.r_[X,d[b'data'].reshape(10000,3,32,32)]
				y = np.r_[y,np.array(d[b'labels'])]
		else:
			p = join(path,'test_batch')
			with open(p,'rb') as f:
				d = pickle.load(f,encoding='bytes')
			X = np.r_[X,d[b'data'].astype(np.float32).reshape(10000,3,32,32)]
			y = np.r_[y,np.array(d[b'labels'])]

		#check size of dataset
		if kind=='train':
			assert X.shape==(50000,3,32,32),'Data missed partially, expect (50000,3,32,32), but got '+str(X.shape)+' instead'
			assert y.shape==(50000,),'Data missed partially, expect (50000,), but got '+str(y.shape)+' instead'
		else:
			assert X.shape==(10000,3,32,32),'Data missed partially, expect (10000,3,32,32), but got '+str(X.shape)+' instead'
			assert y.shape==(10000,),'Data missed partially, expect (10000,), but got '+str(y.shape)+' instead'

		#convert all to tensors
		self.X = torch.from_numpy(X.astype(np.float32))
		self.y = torch.from_numpy(y.astype(np.int64))

	def __getitem__(self,index):
		return self.X[index],self.y[index]

	def __len__(self):
		return self.y.size(0)

class Svhn2Dataset(Dataset):
	def __init__(self,path,kind):

		#read dataset and check size of it
		if kind=='train':
			data = loadmat(join(path,'train_32x32.mat'))
			X = data['X'].transpose((3,2,0,1))
			y = data['y'].reshape(73257)

			assert X.shape==(73257,3,32,32),'Data missed partially, expect (73257,3,32,32), but got '+str(X.shape)+' instead'
			assert y.shape==(73257,),'Data missed partially, expect (73257,), but got '+str(y.shape)+' instead'
		else:
			data = loadmat(join(path,'test_32x32.mat'))
			X = data['X'].transpose((3,2,0,1))
			y = data['y'].reshape(26032)

			assert X.shape==(26032,3,32,32),'Data missed partially, expect (26032,3,32,32), but got '+str(X.shape)+' instead'
			assert y.shape==(26032,),'Data missed partially, expect (26032,), but got '+str(y.shape)+' instead'

		#convert label "10" to "0"
		y = np.where(y==10,0,y)

		#convert all to tensors
		self.X = torch.from_numpy(X.astype(np.float32))
		self.y = torch.from_numpy(y.astype(np.int64))

	def __getitem__(self,index):
		return self.X[index],self.y[index]

	def __len__(self):
		return self.y.size(0)

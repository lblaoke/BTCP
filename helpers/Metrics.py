from sklearn import metrics
import torch

#standard identical-count statistics
def accuracy(y_pred,y_true):
	assert len(y_pred)==len(y_true),'Inconsistent length, %d != %d' %(len(y_pred),len(y_true))

	y_result = torch.where(y_pred>0.5,1,0)
	return metrics.accuracy_score(y_pred=y_result,y_true=y_true)

#balanced accuracy (sensitive to false samples labeled with a minor class)
def balanced_accuracy(y_pred,y_true):
	assert len(y_pred)==len(y_true),'Inconsistent length, %d != %d' %(len(y_pred),len(y_true))

	y_result = torch.where(y_pred>0.5,1,0)
	return metrics.balanced_accuracy_score(y_pred=y_result,y_true=y_true)

#area under PR curve (not yet implemented)
def AUPR(y_pred,y_true):
	assert len(y_pred)==len(y_true),'Inconsistent length, %d != %d' %(len(y_pred),len(y_true))
	return 0.

#area under ROC curve
def AUROC(y_pred,y_true):
	assert len(y_pred)==len(y_true),'Inconsistent length, %d != %d' %(len(y_pred),len(y_true))
	return metrics.roc_auc_score(y_score=y_pred,y_true=y_true)

#Expected Calibration Error
def ECE(y_pred,y_true,K=10):
	assert len(y_pred)==len(y_true),'Inconsistent length, %d != %d' %(len(y_pred),len(y_true))

	index = y_pred.argsort(axis=0)
	bin_conf,bin_y_true = [],[]
	ECE_score = 0.
	bin_len,bin_start = 1./K,0.

	for i in index:
		if y_pred[i]>bin_start+bin_len:
			assert len(bin_conf)==len(bin_y_true),'Bin %d is inconsistent!' % int(bin_start*K)

			if len(bin_conf):
				bin_y_true_tensor = torch.tensor(bin_y_true,dtype=torch.float32)
				e = bin_start+bin_len/2.
				o = bin_y_true_tensor.sum().item()/len(bin_y_true)
				ECE_score += len(bin_conf)/len(y_pred)*abs(e-o)

			del bin_conf,bin_y_true
			bin_conf,bin_y_true = [],[]
			bin_start += bin_len

		bin_conf.append(y_pred[i].item())
		bin_y_true.append(y_true[i].item())

	return ECE_score

#False positive rate (for binary classification only)
def FPR(y_pred,y_true):
	assert len(y_pred)==len(y_true),'Inconsistent length, %d != %d' %(len(y_pred),len(y_true))

	negative_labeled = y_pred[~y_true.bool()]
	false_positive = negative_labeled[negative_labeled>0.5]
	return len(false_positive)/len(y_pred)

#test
if __name__=='__main__':
	a = torch.tensor([[0],[0],[1],[1]])
	b = torch.tensor([[0],[1],[1],[1]])
	print(FPR(a,b))

import torch

class Focal_Loss():
	"""
	二分类Focal Loss
	"""
	def __init__(self,alpha=0.25,gamma=2):
		super(Focal_Loss,self).__init__()
		self.alpha=alpha
		self.gamma=gamma

	def __call__(self,preds,labels):
		"""
		preds:sigmoid的输出结果
		labels：标签
		"""
		eps=1e-7
		loss_1=-1*self.alpha*torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
		loss_0=-1*(1-self.alpha)*torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
		loss=loss_0+loss_1
		return torch.mean(loss)

import numpy as np


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count




def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])
	
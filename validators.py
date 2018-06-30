import os
from PIL import Image
import torch

def checkImage(path):
	try:
		Image.open(path)
	except IOError:
		return False
	return True

def imageValidator(image):
	return checkImage(image)

def pathValidator(path):
	return os.path.exists(path)

def learningRateValidator(lrate):
	return lrate.replace(".", "", 1).isdigit()

def batchSizeValidator(batchsize):
	return batchsize.isdigit()

def epochValidator(epoch):
	return epoch.isdigit()

def workersValidator(workers):
	return workers.isdigit()

def modelCheckpointValidator(model):
	return os.path.exists(model)

def cudaValidator():
	return torch.cuda.is_available()
import os, platform
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
	if platform.system() == "Windows" or platform.system() == "win32":
		path = os.path.abspath(".") + "\\" + path + "\\.."
	else:
		path = os.path.abspath(".") + "/" + path + "/.."
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
	if platform.system() == "Windows" or platform.system() == "win32":
		model = os.path.abspath(".") + "\\" + model
	else:
		model = os.path.abspath(".") + "/" + model
	return os.path.exists(model)

def cudaValidator():
	return torch.cuda.is_available()
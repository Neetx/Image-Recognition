import argparse
from validators import (
	imageValidator,
	learningRateValidator,
	momentumValidator,
	batchSizeValidator,
	epochValidator,
	pathValidator
	)

def argvcontrol():
	parser = argparse.ArgumentParser(description='Image recognition tool implemented with PyTorch, CIFAR10 dataset.', epilog="Ex: sudo ./SshFailToBanBypass.py wordlist.txt -i 127.0.0.1 -p 22 -a 3 -u root")
	parser.add_argument("-i", "--image", help="Image to recognize")
	parser.add_argument("-l", "--learning-rate", help="Learing Rate", default="0.001")
	parser.add_argument("-m", "--momentum", help="Momentum", default="0.9")
	parser.add_argument("-b", "--batch-size", help="Mini-Batch Size", default="4")
	parser.add_argument("-e", "--epoch", help="Number of epoches for training the network", default="3")
	parser.add_argument("-p", "--path", help="Path to read or write the dataset", default="./data")
	args = parser.parse_args()

	valid = True
	if args.image and not imageValidator(args.image):
		print ("[!] Invalid Image")
		valid = False
	if not learningRateValidator(args.learning_rate):
		print ("[!] Invalid Learing Rate")
		valid = False
	if not momentumValidator(args.momentum):
		print ("[!] Invalid Momentum")
		valid = False
	if not batchSizeValidator(args.batch_size):
		print ("[!] Invalid batch-size")
		valid = False
	if not epochValidator(args.epoch):
		print ("[!] Invalid Epoch")
		valid = False
	if not pathValidator(args.path):
		print ("[!] Invalid Path")
		valid = False
	return args, valid
import argparse
from validators import (
	imageValidator,
	learningRateValidator,
	batchSizeValidator,
	epochValidator,
	pathValidator,
	workersValidator,
	modelCheckpointValidator
	)

def argvcontrol():
	parser = argparse.ArgumentParser(description='Image recognition tool implemented with PyTorch, CIFAR10 dataset.', epilog="Application: python image-recognizer.py --image")
	parser.add_argument("-i", "--image", help="Image to recognize")
	parser.add_argument("-m", "--model", help="Model checkpoint file to load/save", default="model_100_76p")
	parser.add_argument("-l", "--learning-rate", help="Learing Rate", default="0.0009")
	parser.add_argument("-b", "--batch-size", help="Mini-Batch Size", default="4")
	parser.add_argument("-e", "--epochs", help="Number of epoches for training the network", default="5")
	parser.add_argument("-p", "--path", help="Path to read or write the dataset", default="./data")
	parser.add_argument("-w", "--workers", help="Number of workers", default="2")
	parser.add_argument('--train', dest='training', help="Set to train and validate the cnn", action='store_true')
	parser.add_argument('--no-train', dest='training',help="Set to perform validation", action='store_false')
	parser.set_defaults(training=False)
	args = parser.parse_args()

	valid = True
	if args.image and not imageValidator(args.image):
		print ("[!] Invalid image")
		valid = False
	if not learningRateValidator(args.learning_rate):
		print ("[!] Invalid learing rate")
		valid = False
	if not batchSizeValidator(args.batch_size):
		print ("[!] Invalid batch-size")
		valid = False
	if not epochValidator(args.epochs):
		print ("[!] Invalid epoch")
		valid = False
	if not pathValidator(args.path):
		print ("[!] Invalid dataset path")
		valid = False
	if not workersValidator(args.workers):
		print ("[!] Invalid workers")
		valid = False
	if not modelCheckpointValidator(args.model):
		print ("[!] Invalid model checkpoint file")
		valid = False
	return args, valid
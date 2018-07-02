from argparser import argvcontrol
import os, sys, platform
from cnn import (
	Net,
	recognition,
	loadModel,
	getTransformtions,
	validation,
	CIFAR10Init,
	training
	)

def main():
	try:
		args, check= argvcontrol()

		if check:
			if platform.system() == "Windows" or platform.system() == "win32":
				args.model = os.path.abspath(".") + "\\" + args.model
				args.path = os.path.abspath(".") + "\\" + args.path
			else:
				args.model = os.path.abspath(".") + "/" + args.model
				args.path = os.path.abspath(".") + "/" + args.path

			classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

			net = Net()
			if args.cuda:
				net.cuda()

			if args.model and args.training:
				if not os.path.exists(args.model):
					print("Model will be created")
				else:
					loadModel(net, args.model)
			elif args.model and not args.training:
				if os.path.exists(args.model):
					loadModel(net, args.model)
				else:
					print("Invalid model checkpoint file. Please enter a valid path.")

			test_transform, train_transform = getTransformtions()

			if args.training:
				trainloader, testloader, criterion = CIFAR10Init(args.cuda, args.path, int(args.batch_size), int(args.workers))
				if trainloader and testloader:
					frequency = int(len(trainloader)/4)
					training(net, args.cuda, int(args.epochs), trainloader, frequency, criterion, float(args.learning_rate), int(args.batch_size), int(args.workers), args.model)
				else:
					exit()

			elif args.image:
				_class = recognition(net, args.image, test_transform, classes)
				if _class:
					print("\nClassification: %s\n" % (_class))

			else:
				trainloader, testloader, criterion = CIFAR10Init(args.cuda, args.path, int(args.batch_size), int(args.workers))
				if trainloader and testloader:
					validation(net, args.cuda, testloader, classes, args.model, int(args.batch_size))
				else:
					exit()
		else:
			print ("\nTraining: python image-recognizer.py --train")
			print ("Validation: python image-recognizer.py --no-train")
			print ("Recognition: python image-recognizer.py --image PATH_TO_IMAGE")
			print ("You can set parameters for this operations. Add --help for more informations.\n")

	except (KeyboardInterrupt, SystemExit):
		pass

if __name__ == "__main__":
	main()
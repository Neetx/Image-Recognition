from argparser import argvcontrol
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
	args, check= argvcontrol()

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	net = Net()
	if args.checkpoint:
		loadModel(net, args.checkpoint)

	test_transform, train_transform = getTransformtions()

	if check:
		if args.training:
			trainloader, testloader, criterion= CIFAR10Init()
			training(net, int(args.epochs), trainloader, 2000, criterion, float(args.learning_rate))

		elif args.image:
			_class = recognition(net, args.image, test_transform, classes)
			print("Classification: %s" % (_class))
		else:
			trainloader, testloader, criterion = CIFAR10Init()
			validation(net, testloader, classes, int(args.batch_size))
	else:
		print ("Usage: blabla")

if __name__ == "__main__":
	main()
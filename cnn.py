import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import torchvision
import torch.optim as optim

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 30, 5, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.bn1 = nn.BatchNorm2d(30)
		self.conv2 = nn.Conv2d(30, 64, 5, padding=1)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 16, 5, padding=1)
		self.bn3 = nn.BatchNorm2d(16)
		self.fc1 = nn.Linear(16 * 2 * 2, 100)
		self.fc2 = nn.Linear(100, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.bn1(x)
		x = self.pool(F.relu(self.conv2(x)))
		x = self.bn2(x)
		x = self.pool(F.relu(self.conv3(x)))
		x = self.bn3(x)
		x = x.view(-1, 16 * 2 * 2)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def getTransformtions():

	test_transform = transforms.Compose(
		[transforms.Resize((32,32)),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
		])
	train_transform = transforms.Compose(
		[transforms.Resize((32,32)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
		])
	return test_transform, train_transform

def CIFAR10Init(cuda, batch_size=4, num_workers=2):
	test_transform, train_transform = getTransformtions()
	print("\n------> Preparing DATASET and DATALOADER")
	try:
		trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	except Exception as e:
		print("[!] Error in CIFAR10 dataset initialization")
		print(e)
		trainloader = None
		testloader = None
	if cuda:
		criterion = nn.CrossEntropyLoss().cuda()
	else:
		criterion = nn.CrossEntropyLoss()

	return trainloader, testloader, criterion

def loadModel(net, modelPath):
	print("\n------> Loading model")
	try:
		net.load_state_dict(torch.load(modelPath))
	except Exception as e:
		print("[!] Error loading model")

def saveModel(net, modelPath):
	print("\n------> Saving model")
	try:
		state = net.state_dict()
		torch.save(state, modelPath)
	except Exception as e:
		print("[!] Error saving model")

def training(net, cuda, epochs, trainloader, frequency, criterion, learning_rate, batch_size, workers, model):
	try:
		optimizer = optim.Adam(net.parameters(), lr=learning_rate,  betas=(0.9, 0.999), eps= 0.09)
	except Exception as e:
		print("[!] Error in optimizer initialization")
		print(e)
		exit()

	print("\n------> Starting training")
	print("------------------------------------------------------------------------")
	print("Training Set length: %d\t Epochs: %d\t Output frequency: %d" % (len(trainloader), epochs, frequency))
	print("Learning rate %.10f\t Batch size: %d\t Workers: %d" % (learning_rate, batch_size, workers))
	print("Model: %s" % (model))
	try:
		for epoch in range(epochs):
			print("------------------------------------------------------------------------")
			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				inputs, labels = data
				if cuda:
					inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda(async=True)
				else:
					inputs, labels = Variable(inputs), Variable(labels)

				optimizer.zero_grad()
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				running_loss += loss.item()
				if i % frequency == frequency-1:
					print('Epoch %d (data %5d) -----> loss: %.3f' % (epoch + 1, i + 1, running_loss/frequency))
					running_loss = 0.0
		print("\nTraining completed.")
	except Exception as e:
		print("[!] Error in training")
		print(e)
		exit()
	saveModel(net, model)

def validation(net, cuda, testloader, classes, model, batch_size=4):
	correct = 0
	total = 0
	print("\n------> Starting total evaluation")
	print("------------------------------------------------------------------------")
	print("Test Set length: %d\t" % ( len(testloader)))
	#print("Learning rate %.10f\t Batch size: %d\t Workers: %d" % (learning_rate, batch_size, workers))
	print("Model: %s" % (model))
	print("------------------------------------------------------------------------")
	try:
		for data in testloader:
			images, labels = data
			if cuda:
				labels = Variable(labels).cuda()
				images = Variable(images).cuda()
			else:
				labels = Variable(labels)
				images = Variable(images)
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
		print("Accuracy: %d %%" % (100 * correct / total))
	except Exception as e:
		print("[!] Error in total evaluation")
		print(e)

	print("\n------> Starting classes evaluation..")
	try:
		class_correct = list(0. for i in range(10))
		class_total = list(0. for i in range(10))
		for data in testloader:
			images, labels = data
			if cuda:
				labels = Variable(labels).cuda()
				images = Variable(images).cuda()
			else:
				labels = Variable(labels)
				images = Variable(images)
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			c = (predicted == labels).squeeze()
			for i in range(batch_size):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

		for i in range(10):
			print("Accuracy of %5s: %2d %%" % (classes[i], 100 * class_correct[i]/class_total[i]))
	except Exception as e:
		print("[!] Error in classes evaluation")
		print(e)

def recognition(net, path, transform, classes):
	out = ""
	try:
		image = Image.open(path)
		image = image.convert("RGB")
		image_tensor = transform(image).unsqueeze_(0)
		image_var = Variable(image_tensor)
		outputs = net(image_var)
		_, predicted = torch.max(outputs.data, 1)
		out = classes[predicted[0]]
	except Exception as e:
		print("[!] Error in image recognition")
		print(e)
		out = None

	return out
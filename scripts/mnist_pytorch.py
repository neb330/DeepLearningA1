from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

print('loading data!')
path = '../../data/'
#trainset_labeled = pickle.load(open(path + "train_labeled.p", "rb"))

#augmented data
trainset_labeled = pickle.load(open(path + "trainset_new.p", "rb"))
##
#validset = pickle.load(open(path + "validation.p", "rb"))
##trainset_unlabeled = pickle.load(open(path + "train_unlabeled.p", "rb"))
##print('done')
##
train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64, shuffle=True, **kwargs)
#valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)
#unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=64, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize(
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=2)
#        self.conv1 = nn.Conv2d(1, 8, kernel_size=4)
#        self.conv2 = nn.Conv2d(8, 16, kernel_size=4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print ('x', x.size())
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

def train(epoch):
    model.train()
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
#        print ('target size', target.size())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    #store values to plot accuracy
    train_accuracy.append(100. * correct/len(train_loader.dataset))
            
def test(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    
    #store values to plot accuracy
    test_accuracy.append(100. * correct / len(valid_loader.dataset))

def plot_accuracy(epochs, accuracy_test, accuracy_train):
    plt.plot(epochs, accuracy_test, label = 'Validation')
    plt.plot(epochs, accuracy_train, label = 'Train')
    plt.title("Validation and Train accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fancybox=True, shadow=True, ncol=5)
    plt.show()

def plot_accuracy_zoomin(epochs, accuracy_test, accuracy_train):
    plt.plot(epochs, accuracy_test, label = 'Validation')
    plt.plot(epochs, accuracy_train, label = 'Train')
    plt.title("Validation and Train accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim((90,100))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fancybox=True, shadow=True, ncol=5)
    plt.show()    
    
def plot_accuracy_grid(epochs, accuracy, parameter_values, parameter_name, title, plot_train=False):
    for i in range(len(parameter_values)):
        if plot_train:
            plt.plot(epochs, accuracy[i][0], label = 'Trainig data ' + 
                 parameter_name[i] + ' ' + str(parameter_values[i]))
        plt.plot(epochs, accuracy[i][1], label = parameter_name + ' ' + str(parameter_values[i]))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True, ncol=1)
    plt.title(title)
    plt.show()
    
#for testing
args.epochs = 10
args.lr = 0.003
args.momentum = 0.95
model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
print ('args.lr', args.lr)

train_accuracy = []
train_epoch = []
test_accuracy = []
test_epoch = []
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch, valid_loader)
plot_accuracy(range(1, args.epochs + 1), test_accuracy, train_accuracy)
plot_accuracy_zoomin(range(1, args.epochs + 1), test_accuracy, train_accuracy)


#testset = pickle.load(open(path + "test.p", "rb"))
#test_loader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False)

print (test(1, test_loader))


#momentum = [0.5, 0.9, 0.95, 0.99]
#args.momentum = 0.95
#train_accuracy = []
#train_epoch = []
#test_accuracy = []
#test_epoch = []
#accuracy = []
#
#model = Net()
#print ('args.lr', args.lr)
#
#for i in range(1):
#    model = Net()
#    args.epochs = 1
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
##    optimizer = optim.nag(model.parameters(), lr=args.lr, momentum=args.momentum)
#    print ('args.lr', args.lr)
#    
#    train_accuracy = []
#    train_epoch = []
#    test_accuracy = []
#    test_epoch = []
#    for epoch in range(1, args.epochs + 1):
#        train(epoch)
#        test(epoch, valid_loader)
#    accuracy.append([train_accuracy, test_accuracy])
#plot_accuracy_grid(range(1, args.epochs + 1), accuracy, num_epochs, 'Number of Epochs', 
#'Accuracy by number of epochs (validation set)', False)



    

    
    
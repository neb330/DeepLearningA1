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
import math



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
path = './'
trainset_labeled = pickle.load(open(path + "train_labeled.p", "rb"))
validset = pickle.load(open(path + "validation.p", "rb"))
trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))
print("labeled set size", trainset_labeled)


# print("type of unlabeled data")
# print (type(trainset_unlabeled))
# print('done')
train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)
train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=64, shuffle=True, **kwargs)

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)



def weightingFunction(epoch,T1=10.1,T2=30,alpha =.25):
    if epoch<T1:
        return 0
    elif (epoch>=T1) and (epoch<=T2):
        return alpha*( T1 - epoch)/(T1-T2)
    else :
        return alpha

    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, 80)

        self.fc3 = nn.Linear(80, 10)

        # print ("modules", self.modules())
        # for m in self.modules():
        #     print("module", m)
        #     if isinstance(m, nn.Linear):
        #         size = m.weight.size()
        #         fan_out = size[0] # number of rows
        #         fan_in = size[1] # number of columns
        #         n = fan_out*fan_in
        #         m.weight.data.uniform_(-.01, .01)
            # if isinstance(m, nn.Conv2d):
            #     n = m.size()
            #     print("module", m)
            #     print("num param ",n)
            #     m.weight.data.uniform_(-math.sqrt(6/(n+1)), math.sqrt(6/(n+1)))
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
print ('args.lr', args.lr)

train_accuracy = []
train_epoch = []
test_accuracy = []
test_epoch = []


def train_unlabeled(epoch):
    model.train()
    correct = 0
    print("epoch of unlabeled, " , epoch)
    print("weighing function" , weightingFunction(epoch))
    lastTestAccuracy = test_accuracy[-1]

    if weightingFunction(epoch)==0:
        return  

    for batch_idx, (data, target) in enumerate(train_unlabeled_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data)

        optimizer.zero_grad()
        output = model(data)

        # print ("current best prediction")
        target = output.data.max(1)[1]
        target = target.view(target.size()[0]) #make 1d array
        target = Variable(target)



        loss = weightingFunction(epoch)*F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch unlabled: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_unlabeled_loader.dataset),
        #         100. * batch_idx / len(train_unlabeled_loader), loss.data[0]))
        
        # pred = output.data.max(1)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data).cpu().sum()

    #store values to plot accuracy
    # train_accuracy.append(100. * correct/len(train_loader.dataset))
    # print ("new train accuracy", 100. * correct/float(len(train_loader.dataset)))
            

def train(epoch):
    model.train()
    correct = 0

    print("epoch of unlabeled")

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # print("target according to train", target)
        # print (type(target))
        data, target = Variable(data), Variable(target)
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
    print ("new train accuracy", 100. * correct/float(len(train_loader.dataset)))
            
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
    test_accuracy.append(100. * correct / float(len(valid_loader.dataset)))

#for testing
args.epochs = 160

for epoch in range(1, args.epochs + 1):
    train(epoch)
    train_unlabeled(epoch)
    test(epoch, valid_loader)



def plot_accuracy(epochs, accuracy_test, accuracy_train):

    plt.plot(epochs, accuracy_test, label = 'Test')
    plt.plot(epochs, accuracy_train, label = 'Train')
    plt.title("Test and Train accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

print ("test")
print (test_accuracy)
print ("train")
print (train_accuracy)
plot_accuracy(range(1, args.epochs + 1), test_accuracy, train_accuracy)
    
    
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
from sklearn.preprocessing import MinMaxScaler


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def load_data():
    data_train = list(open('../data/vae_encodings.tr').readlines())
    targets_train = list(open('../data/vae_targets.tr').readlines())
    data_val = list(open('../data/vae_encodings.val').readlines())
    targets_val = list(open('../data/vae_targets.val').readlines())

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for vec, label in zip(data_train, targets_train):
        vec = vec.strip().split()
        vec = [float(num) for num in vec]
        label = label.strip()
        X_train.append(vec)
        y_train.append(float(label))
    for vec, label in zip(data_val, targets_val):
        vec = vec.strip().split()
        vec = [float(num) for num in vec]
        label = label.strip()
        X_val.append(vec)
        y_val.append(float(label))
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(np.array(X_train))
    X_val = min_max_scaler.fit_transform(np.array(X_val))
    return [X_train, y_train, X_val, y_val]




def batch_iter(data, targets, batch_size, num_epochs, shuffle=True):
    """
        Generates a batch iterator for a dataset.
        """
    targets = np.array(targets)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        shuffled_targets = targets[shuffle_indices]
    else:
        shuffled_data = data
        shuffled_targets = targets
    for batch_num in range(num_batches_per_epoch - 1):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield [shuffled_data[start_index:end_index], shuffled_targets[start_index:end_index]]

#for testing
args.epochs = 1000
args.batch_size = 50

mb_size = args.batch_size
n_hidden_1 = 350
n_hidden_2 = 350
X_dim = 100
n_classes = 10
c = 0
lr = 1e-3


# Create model
def multilayer_perceptron(x, weights, biases):
    #print(x, weights[0], biases[0])
    
    # Hidden layer with RELU activation
    layer_1 = F.relu(torch.mm(x, weights[0]) + torch.transpose(biases[0].expand(n_hidden_1, mb_size), 0, 1))
    # Hidden layer with RELU activation
    layer_2 = F.relu(torch.mm(layer_1, weights[1]) + torch.transpose(biases[1].expand(n_hidden_2, mb_size), 0, 1))
    # Output layer with linear activation
    out_layer = torch.mm(layer_2, weights[2]) + torch.transpose(biases[2].expand(n_classes, mb_size), 0, 1)
    return F.log_softmax(out_layer)

weights = [Variable(torch.randn((X_dim, n_hidden_1)), requires_grad=True), Variable(torch.randn((n_hidden_1, n_hidden_2)), requires_grad=True), Variable(torch.randn((n_hidden_2, n_classes)), requires_grad=True)]

biases = [Variable(torch.randn((n_hidden_1, 1)), requires_grad=True), Variable(torch.randn((n_hidden_2, 1)), requires_grad=True), Variable(torch.randn((n_classes, 1)), requires_grad=True)]

params = list(weights + biases)

optimizer = optim.Adam(params, lr=args.lr)

train_accuracy = []
train_epoch = []
test_accuracy = []
test_epoch = []

X_train, y_train, X_val, y_val = load_data()

def train(epoch):
    correct = 0
    batches = batch_iter(X_train, y_train, args.batch_size, args.epochs)
    for batch_idx, batch in enumerate(batches):
        data, target = batch
        #data = data.astype(float)
        target = target.astype(int)
        data, target = torch.Tensor(data), torch.LongTensor(target)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = multilayer_perceptron(data, weights, biases)
        loss = F.nll_loss(output, target)
        '''m = nn.CrossEntropyLoss()
        loss = m(output, target)'''
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss.data[0]))
    
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    #store values to plot accuracy
    train_accuracy.append(correct/(len(y_train)))


def test(epoch):
    test_loss = 0
    correct = 0
    batches = batch_iter(X_val, y_val, args.batch_size, args.epochs)
    for batch_idx, batch in enumerate(batches):
        data, target = batch
        target = target.astype(int)
        data, target = torch.Tensor(data), torch.LongTensor(target)
        data, target = Variable(data), Variable(target)
        output = multilayer_perceptron(data, weights, biases)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(X_val) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy:({:.0f}%)\n'.format(test_loss, 100. * correct / (len(y_val))))
    
    #store values to plot accuracy
    test_accuracy.append(100. * (correct/(10*batch_idx)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)


'''def plot_accuracy(epochs, accuracy_test, accuracy_train):
    plt.plot(epochs, accuracy_test, label = 'Test')
    plt.plot(epochs, accuracy_train, label = 'Train')
    plt.title("Test and Train accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
plot_accuracy(range(1, args.epochs + 1), test_accuracy, train_accuracy)'''
    
    
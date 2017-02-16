from __future__ import print_function
import torch
import torch.nn.functional as nn
import torchvision.transforms as transforms
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pickle
import argparse
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
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


print('loading data!')
path = '../data/'
trainset_labeled = pickle.load(open(path + "train_labeled.p", "rb"))
#trainset_unlabeled = pickle.load(open(path + "train_unlabeled.p", "rb"))
train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=10, shuffle=True, **kwargs)
train_final_load = torch.utils.data.DataLoader(trainset_labeled, batch_size=10, shuffle=True, **kwargs)
print('done')



#mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 10
Z_dim = 100
X_dim = 784
y_dim = 10
h_dim = 128
c = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


# =============================== Q(z|X) ======================================

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(mb_size, h_dim), requires_grad=True)

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(mb_size, Z_dim), requires_grad=True)

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(mb_size, Z_dim), requires_grad=True)


def Q(X):
    h = nn.relu( torch.mm(X, Wxh) + bxh)
    z_mu = torch.mm(h, Whz_mu) + bhz_mu
    z_var = torch.mm(h, Whz_var) + bhz_var
    return z_mu, z_var


def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(mb_size, h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(mb_size, X_dim), requires_grad=True)


def P(z):
    h = nn.relu(torch.mm(z, Wzh) + bzh)
    X = nn.sigmoid(torch.mm(h,Whx) + bhx)
    return X


# =============================== TRAINING ====================================

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
          Wzh, bzh, Whx, bhx]

solver = optim.Adam(params, lr=lr)

for batch_idx, (data, target) in enumerate(train_loader):
    X = data.view(mb_size, 784)
    X = Variable(X)
    
    # Forward
    z_mu, z_var = Q(X)
    z = sample_z(z_mu, z_var)
    X_sample = P(z)

    # Loss
    recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = recon_loss + kl_loss
    
    # Backward
    loss.backward()
    
    # Update
    solver.step()
    
    # Housekeeping
    for p in params:
        p.grad.data.zero_()
    
    # Print and plot every now and then
    if batch_idx % 1000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(batch_idx, loss.data[0]))
        
        samples = P(z).data.numpy()[:16]
        
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
        
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        
        if not os.path.exists('out/'):
            os.makedirs('out/')
        
        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
        plt.close(fig)

arr = np.zeros((3000, Z_dim))
for batch_idx, (data, target) in enumerate(train_final_load):
    X = data.view(mb_size, 784)
    X = Variable(X)
    
    # Forward
    z_mu, z_var = Q(X)
    z = sample_z(z_mu, z_var)
    #new_z = z.asNumpyTensor()
    print(new_z.shape)
#np.append(arr, np.array(z), axis=0)






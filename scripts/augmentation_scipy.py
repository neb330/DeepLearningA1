import numpy as np
import pickle
import torch
from torchvision import transforms
from scipy import ndimage
from matplotlib import pyplot as plt
from sub import subMNIST 

def rotate(image, degree):
    return ndimage.rotate(image, degree, reshape=False)

def gaus_filter(image, sigma=1.5):
    return ndimage.gaussian_filter(image, sigma)

print('loading data!')
path = '../../data/'
trainset_labeled = pickle.load(open(path + "train_labeled.p", "rb"))
print('done')
train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=3000, shuffle=True)
data = train_loader.dataset.train_data
data_images = data.numpy()
target = train_loader.dataset.train_labels
target = target.numpy()

#crop at random to get 24*24 images

image = data_images[0]
new_images = []
new_labels = []

for i in xrange(data_images.shape[0]):
    created_images = []
    image = data_images[i]
    created_images.append(rotate(image, 0))
    created_images.append(rotate(image, 30))
    created_images.append(rotate(image, -30))
    created_images.append(gaus_filter(image))
    new_images.extend(np.array(created_images))
    new_labels.extend([target[i]] * len(created_images))

    #    if i % 300 == 0:
#        print 'original'
#        plt.imshow(image)
#        plt.show()
#        for im in created_images:
#            plt.imshow(im)
#            plt.show()
    
size = len(new_images)
print 'new size:', size
  
new_train_data = np.array(new_images)
new_train_labels = np.array(new_labels)
print 'images added:', size - 3000
print 'labels:', new_train_labels.shape

new_train_data = torch.from_numpy(new_train_data)
new_train_labels = torch.from_numpy(new_train_labels)


transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                             ])
trainset_new = subMNIST(root=path + 'data', train=True, transform=transform, download=True, k=size)
trainset_new.train_data = new_train_data.clone()
trainset_new.train_labels = new_train_labels.clone()   

pickle.dump(trainset_new, open(path + "trainset_new.p", "wb" ))
    
    



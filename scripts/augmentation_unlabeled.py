import numpy as np
import pickle
import torch
from torchvision import transforms
from sub import subMNIST 
import augmentation_scipy as a
    
print('loading data!')
path = '../../data/'
trainset_unlabeled = pickle.load(open(path + "train_unlabeled_Mike.p", "rb"))
train_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=47000, shuffle=True)
data = train_loader.dataset.train_data
data_images = data.numpy()
target = train_loader.dataset.train_labels
target = target.numpy()
print('done')

image = data_images[0]
new_images = []

for i in xrange(data_images.shape[0]):
    created_images = []
    image = data_images[i]
    created_images.append(a.rotate(image, 0))        
    for j in range(3):
        created_images.append(a.elastic_transform(image, 34, 4))
    new_images.extend(np.array(created_images))
    if i % 1000 == 0:
        print ('done with ', i)
   
size = len(new_images)
print 'new size:', size
  
new_train_data = np.array(new_images)
print 'images added:', size - 47000

new_train_data = torch.from_numpy(new_train_data)
new_train_labels = torch.from_numpy(np.array([-1] * size))


transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                             ])
trainset_new = subMNIST(root=path + 'data', train=True, transform=transform, download=True, k=size)
trainset_new.train_data = new_train_data.clone()
trainset_new.train_labels = new_train_labels.clone()   

pickle.dump(trainset_new, open(path + "unlab_trainset_new.p", "wb" ))
    
    



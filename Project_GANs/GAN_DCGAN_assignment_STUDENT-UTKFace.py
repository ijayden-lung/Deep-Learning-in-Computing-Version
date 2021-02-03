#!/usr/bin/env python
# coding: utf-8

# Project 4: Generative Adversarial Networks
# ======
# In this project, you are expected to fill in the missing parts of a PyTorch implementation of the Deep Convolutional Generative Adversarial Network (DCGAN) and test its performance on several datasets.
# 
# We will be using datasets with small images for this project, because high-resolution GANs take notoriously long to train.
# 
# **References:**
# 
# [1] [Radford, A., Metz, L. and Chintala, S., 2015. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.](https://arxiv.org/abs/1511.06434)
# 
# [2] [MNIST dataset.](http://yann.lecun.com/exdb/mnist/)
# 
# [3] [FashionMNIST dataset.](https://github.com/zalandoresearch/fashion-mnist)
# 

# ## Assignment 4A: DCGAN

# ### Verify your PyTorch installation

# In[1]:


import torch
import torchvision

# Print Basic Information
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print("Torchvision", torchvision.__version__)
print('Device:', torch.device('cuda:0'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


#import statements
import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from torchsummary import summary
import numpy as np


# In[3]:




# In[4]:


#######################################################################################################
# TODO 2: choose sensible training parameters for DCGAN training. 
# Start with few epochs while you design the network, and increase once you think you have a good setup.
# Since GANs train slowly, you may have to use at quite a few epochs to see good results.
#######################################################################################################
lr = 0.0002
num_epochs = 20
batch_size = 128
workers = 8 # The number of parallel processes used to read data
gpu_id = [0,1,2,3,4,5,6,7] # only modify if you machine has more than one GPU card

nz = 100 #length of latent vector
ngf = 64 #relates to the depth of feature maps carried through the generator
ndf = 64 # Size of feature maps in discriminator
nc = 3  #number of color channels in the input images.

            


# ### 7) Modify the network to output $64\times64$ RGB images (10 points)
# 
# Download `UTK_face.tar.gz` from [Google Drive](https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE) and extract the images to your working directory in the folder `./data/UTKFace/train`. This dataset contains approximately 24K cropped images of human faces of mixed age, gender and race.
# 
# Load these images and resize them to the dimensions you want to generate ($64\times64$ RGB images). Replicate and modify your Generator and Discriminator network architectures in order to output the appropriate image size and three color channels. You may want to consult the [original DCGAN paper](https://arxiv.org/abs/1511.06434) to check what their suggested architecture looks like. You should generate an output similar to this (or better):
# 
# <img src="img/DCGAN_faces.png" width="900">
# 
# 
# 
# You can either modify the code above, or copy the necessary code pieces, but make sure you include all results of different datasets in your final report or in this notebook.

# In[14]:


#######################################################################################################
# TODO 7: resize the images such that they have the shape of 3x64x64
#######################################################################################################
transform = transforms.Compose([
                    transforms.Resize(64),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
    # the pictures in minist dataset is grayscaled, which means there is only one sigle channel
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


train_dataset = torchvision.datasets.ImageFolder(
    root='./UTKFace',
    transform=transform
)

print("Dataset length: ", len(train_dataset))
print("Image size: {}".format(train_dataset[0][0].size()))
print("value range: [ {} - {} ]".format(torch.min(train_dataset[0][0]), torch.max(train_dataset[0][0])))


# In[ ]:


#######################################################################################################
# TODO 7: train a DCGAN on the face images
#######################################################################################################


# In[5]:


#######################################################################################################
# TODO 3a: implement a function that initializes the layer's weights with mean and standard deviation.
#######################################################################################################

def normal_init(layer, mean, std):
    #pass # TODO add your implementation
    nn.init.normal_(layer.weight.data, mean, std)
    


# In[18]:


#######################################################################################################
# TODO 3b: implement the generator network for a DCGAN with output size 32x32px
#######################################################################################################
#The job of the generator is to spawn ‘fake’ images that look like the training images. 
#The job of the discriminator is to look at an image and output whether or not it is a real training image 
#or a fake image from the generator. 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #pass # TODO add your implementation
        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2,ngf*1,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*1),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf*1, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    # forward method
    def forward(self, x):
        #return None # TODO add your implementation
        return self.generator(x)
        


# In[19]:


#######################################################################################################
# TODO 3c: implement the discriminator network for a DCGAN with input size 32x32px
#######################################################################################################
# batch norm and leaky relu functions promote healthy gradient flow
#which is critical for the learning process of both G and D.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #pass # TODO add your implementation
        self.discriminator = nn.Sequential(
            # input is (nc) x 32x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    # forward method
    def forward(self, x):
        #return None # TODO add your implementation
        return self.discriminator(x)
    


# In[20]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    #elif classname.find('BatchNorm') != -1:
     #   nn.init.normal_(m.weight.data, 1.0, 0.02)
      #  nn.init.constant_(m.bias.data, 0)


# In[21]:


grid_size = 6

#######################################################################################################
# TODO 4: generate a fixed noise vector to repeatedly evaluate the output of the generator 
# with the same noise vector as training progresses
#######################################################################################################
fixed_z = torch.randn(grid_size**2, nz, 1, 1, device=device)

# output generated samples from the current state of the generator network
def show_result(num_epoch, show=False, save=False, grid_size=6, path = 'result.png', useFixed=False):
    with torch.no_grad():
        if useFixed:   
            fake = G(fixed_z).cpu()
        else:
            # TODO generate a new random noise vector
            z = torch.randn(36, 100, 1, 1).to(device)
            fake = G(z).cpu()

        fig, ax = plt.subplots(figsize=(8, 8))
        
        image = utils.make_grid(fake.data, grid_size, 1,normalize=True)
        
        plt.imshow(np.transpose(image, (1, 2, 0)))    
        ax.axis('off')
        plt.savefig(path)
        if show:
            plt.show()
        else:
            plt.close()

# plot the losses over time as a curve            
def plot_losses(hist, show=True, save=False, path='progress.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# ### 5) Training Setup (5 points) and Training Loop (10 points)
# 
# Instantiate a Generator and a discriminator network, move networks to GPU and initialize the weights of their convolutional layers with $\mu=0$ and $\sigma=0.02$.
# 
# Create an ADAM optimizer for the generator as well as the discriminator with your chosen learning rate. Use $\beta_1=0.5$ and $\beta_2=0.999$ as described in Section 4 of the paper.
# 
# In the training loop, evaluate the output of the generator network on a random latent vector of length 100. Then, evaluate the performance of the discriminator on the generated images as well as the real image minibatch. Assign the real results the label $1$ (`torch.ones`) and the fake results the label $0$ (`torch.zeros`), then calculate the losses for each of these steps using Binary Cross Entropy loss.
#         
# 
# #### Hint:
# If you would like to load a stored model for additional training or for evaluation, you can do it the following way:
# ```
# M = MyNetwork() #instantiate the network
# checkpoint = torch.load("path_to_stored_network.pkl")
# M.load_state_dict(checkpoint)
# M.eval()
# ``` 
# 
# *If your model fails to train properly, include the failure cases into your report, explain what you think went wrong, and try again!*

# In[22]:


data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=workers)

dset_name = 'myDataset_UTKFace' #pick a meaningful name
result_dir = '{}_DCGAN'.format(dset_name)


# In[23]:


#######################################################################################################
# TODO 5a: Fill code for training setup
#######################################################################################################

# define networks

G = Generator().to(device)
G = nn.DataParallel(G, device_ids=[g for g in gpu_id])
D = Discriminator().to(device)
D = nn.DataParallel(D, device_ids=[g for g in gpu_id])

G.apply(weights_init)
D.apply(weights_init)


# define optimizers
# Initialize BCELoss function
criterion = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


# In[24]:



print(G)
print(D)


# In[25]:


print('Generator')
for name,param in G.named_parameters():
    print(name,torch.std(param))
print('Discriminator')
for name,param in D.named_parameters():
    print(name,torch.mean(param))
    print(name,torch.std(param))


# In[26]:


# results save folder
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
if not os.path.isdir(result_dir+'/Random'):
    os.mkdir(result_dir+'/Random')
if not os.path.isdir(result_dir+'/Fixed'):
    os.mkdir(result_dir+'/Fixed')


# In[27]:


progress = {}
progress['D_losses'] = []
progress['G_losses'] = []
progress['per_epoch_times'] = []
progress['total_time'] = []
num_iter = 0

# start training loop
print('Training ...')
start_time = time.time()
for epoch in np.arange(num_epochs):
    
    D_losses = []
    G_losses = []
    
    epoch_start_time = time.time()
    for real, _ in data_loader:
        real = real.to(device)
        batch_size = real.size(0)
        
        #######################################################################################################
        # TODO 5b: Fill code for training loop
        #######################################################################################################
        
        # generate random latent vector z and real and fake labels 
        z = torch.randn(batch_size, 100, 1, 1).to(device)
        real_label = torch.full((batch_size,), 1).to(device)
        fake_label = torch.full((batch_size,), 0).to(device)
    
        # TODO 
        
        # generate fakes    
        fakes = G(z)

        # TODO 
        
        # evaluate fakes
        err_fakes_D = criterion(D(fakes.detach()).view(-1), fake_label)
        
        # TODO 

        # evaluate real minibatch
        err_real_D = criterion(D(real).view(-1), real_label)
         
        # TODO
        
        # accumulate discriminator loss from the information about the real and fake images it just saw
        D_train_loss = err_fakes_D + err_real_D

        # train discriminator D step
        D.zero_grad()
        D_train_loss.backward()
        D_optimizer.step()    
        
        # train generator to output an image that is classified as real              
        G_train_loss = criterion(D(fakes).view(-1), real_label)
        
        # train generator G step
        G.zero_grad()
        G_train_loss.backward()
        G_optimizer.step()              

        D_losses.append(D_train_loss.data.item())
        G_losses.append(G_train_loss.data.item())

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_time = epoch_end_time - epoch_start_time
    
    print('Epoch [{} / {}] G loss: {} D loss: {}'.format(epoch + 1, num_epochs, G_train_loss, D_train_loss))

    show_result( epoch, save=True, path=result_dir + '/Random/{}_DCGAN_{}.png'.format(dset_name, epoch), useFixed=False )
    show_result( epoch, save=True, show=True, path=result_dir + '/Fixed/{}_DCGAN_{}.png'.format(dset_name, epoch), useFixed=True )
    progress['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    progress['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    progress['per_epoch_times'].append(per_epoch_time)

end_time = time.time()
total_time = end_time - start_time
progress['total_time'].append(total_time)

print('Avg per epoch time: {:.2f} sec, total {:d} epochs time: {:.2f} min'.format(torch.mean(torch.FloatTensor(progress['per_epoch_times'])), num_epochs, total_time / 60))
print('Training finished!')
print('...saving training results')

torch.save(G.state_dict(), result_dir + '/generator_network.pkl')
torch.save(D.state_dict(), result_dir + '/discriminator_network.pkl')
with open(result_dir + '/progress.pkl', 'wb') as f:
    pickle.dump(progress, f)

# plot the loss curves    
plot_losses(progress, save=True, path=result_dir + '/{}_DCGAN_progress.png'.format(dset_name))

# Visualize the training progress as an animated GIF
images = []
for e in range(num_epochs):
    img_name = result_dir + '/Fixed/{}_DCGAN_{}.png'.format(dset_name, e)
    images.append(imageio.imread(img_name))
imageio.mimsave(result_dir + '/DCGAN_generation_animation.gif', images, fps=5)


# ### Display the animated generation GIF
# 
# % edit path to GIF %
# <img src='./myDataset_DCGAN/DCGAN_generation_animation.gif' width="512">

# ### 6) Explore the latent space of the generator (5 points)
# We now want to explore the expressiveness of the Generator. Intuitively, a small change in values of the latent vector $z$ should result in a small change in the output image, and a linear interpolation between two latent vectors $z_1$ and $z_2$ should yield a reasonably smooth transition from one valid output image to another. Implement a function that creates a grid of latent vectors, where each row is an interpolation from the leftmost to the rightmost latent vector, with $n$ interpolation steps.
# 
# The expected output of this function should look like this (or better): 
# 
# <img src="img/latent_space_morph_MNIST.PNG" width="900">
# 
# 
# <img src="img/latent_space_morph_FashionMNIST.png" width="900">

# In[18]:


#generator_path = './myDataset_DCGAN/generator_network.pkl'
#G = Generator().to(device)
#G = nn.DataParallel(G, device_ids=[g for g in gpu_id])
#G.load_state_dict(torch.load(generator_path))
G.eval()


# In[21]:


def generate_latent_interpolation(rows=5, interp_steps=10):
     with torch.no_grad():
        #######################################################################################################
        # TODO 6: for each row, create interpolation between leftmost and rightmost output image
        #######################################################################################################
        #leftmot
        for row in range(rows):
            z_left = torch.randn(1,100,1,1)
            z_s = z_left
            # rightmost
            z_right = torch.randn(1,100,1,1)
            difference = (z_right - z_left)/(interp_steps-1)
            for i in range(1, interp_steps-1):
                z = difference * i + z_left
                z_s = torch.cat((z_s, z), 0)
            z = torch.cat((z_s, z_right), 0).to(device)
            fakes = G(z).cpu()


                # TODO 

            grid_images = fakes.data   

            fig, ax = plt.subplots(figsize=(20, 10))
            grid = utils.make_grid(grid_images, interp_steps)
            grid = 1-grid
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            ax.axis('off')

generate_latent_interpolation()        


# ### 8) Create a latent space walk of the Face Generator you trained (5 points)
# To that end, generate as many random latents as your animation has number of frames. Then, use a gaussian filter on those latents in order to create a smooth blending between adjacent frames in the animation. Have a look at `scipy.ndimage.gaussian_filter` for that purpose. Make sure that your animation can loop endlessly, i.e. the last frames and the first frames of the animation also blend. The $\sigma$ value of the gaussian filter defines how many frames are blended; use the given parameter values of the function to control this value.<br>
# Your output should look like this (or better):
# 
# <img src='img/Faces_DCGAN_latent_walk_1.gif' style='margin: 0px 10px' align='left' width="128">
# 
# <img src='img/Faces_DCGAN_latent_walk_2.gif' style='margin: 0px 10px' align='left' width="128">
# 
# <img src='img/Faces_DCGAN_latent_walk_3.gif' style='margin: 0px 10px' align='left' width="128">

# In[ ]:


def generate_latent_walk(duration_sec=20.0, smoothing_sec=1.0, fps=30):
    #######################################################################################################
    # TODO 8: create a smoothly interpolated loop through random latent space values
    #######################################################################################################
    
    images = None #TODO create smoothly interpolating images
    
    imageio.mimsave(result_dir + '/latent_space_walk.gif', images, fps=fps)

generate_latent_walk()       

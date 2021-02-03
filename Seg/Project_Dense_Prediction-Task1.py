import os
import gc
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--lr',default=None)
opts = parser.parse_args()
learning_rate= float(opts.lr)

epochs = 30
batch_size = 8
#learning_rate = 1
workers = 4 # The number of parallel processes used to read data
gpu_id = [0,1,2,3] # only modify if you machine has more than one GPU card


from loaders import prep_loaders
train_loader, valid_loader = prep_loaders('UnrealData256', batch_size=batch_size, workers=workers)




class densenet(nn.Module):
    def __init__(self):
        super(densenet, self).__init__()
        self.densenet161 = models.densenet161( pretrained=True )

    def forward(self, x):
        downsampled = [x]

        for key, value in self.densenet161.features._modules.items():
            downsampled.append(value(downsampled[-1]))
        return downsampled # return a list
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = Down(in_channels, out_channels,pooling=False)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels,pooling=True):
        super().__init__()
        self.pooling = pooling
        self.maxpool = nn.MaxPool2d(2)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )  

    def forward(self, x):
        if(self.pooling):
            return self.maxpool(self.double_conv(x))  
        else:
            return self.double_conv(x)


class Down2(nn.Module):
    def __init__(self, in_channels, out_channels,pooling=True):
        super().__init__()
        self.pooling = pooling
        self.maxpool = nn.MaxPool2d(2)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )   

    def forward(self, x):
        if(self.pooling):
            return self.maxpool(self.double_conv(x))  
        else:
            return self.double_conv(x)
        

    
class expansive(nn.Module):
    def __init__(self, in_channels, out_channels,length,filter_size=3, padding=0):
        super(expansive, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        #self.conv = double_conv(in_ch//2 + length, out_ch, filter_size=filter_size, padding=padding)
        #self.conv = DoubleConv(in_channels, out_channels)
        self.conv = Down(in_channels//2,out_channels,pooling=False)

    def forward(self, upsampled, downsampled):
        upsampled = self.up(upsampled)

        x = torch.cat((upsampled, downsampled), dim=1)
        x = self.conv(x)
        return x
class VGGUnetModel(nn.Module):
    def __init__(self):
        super(VGGUnetModel, self).__init__()

        
       
        self.down1 = Down(3,64,pooling=False)
        self.down2 = Down(64,128,pooling=True)
        self.down3 = Down2(128,256,pooling=True)
        self.down4 = Down2(256,512,pooling=True)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(64, 1, kernel_size=1)
        
        #self.up1 = expansive(256*2,128*2,96,3,padding=1)
        # (128,44)->(64,88)->(128,88)->padding=0(64,84)->(64,80)
        #self.up2 = expansive(128*2,64*2,96,3,padding=1)
        # (64,80)->(32,160)->(64,160)->padding=0(32,156)->(32,152)
        #self.up3 = expansive(64*2,32*2,3,padding=1)
#         self.resnet = nn.Sequential(*list(resnet18.children())[4:5])
        #self.conv1 = nn.Conv2d(128, 64, 1, padding=0)
        #self.conv2 = nn.Conv2d(64, 1, 1, padding=0)
        #self.get_downsample_features = densenet()
        
        
    def forward(self, x):
        #fea = self.get_downsample_features(x)
        #x1, x2 = fea[3], fea[4]
        #x = self.down1(x) 
        #x = self.down2(x) 
        #x = self.down3(x)
        #x = self.down4(x) 
        #x = self.up1(x, x2)
        #x = self.up2(x, x1)
        #x = self.conv1(x)
        #x = self.conv2(x)
        
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.maxpool(x)
        x = self.conv(x)

        return x

#model = UNet(bilinear=False)
model = VGGUnetModel()
model = model.cuda()
model = nn.DataParallel(model, device_ids=[g for g in gpu_id])


import torch
from math import exp
import torch.nn.functional as F


def loss_fn(pred_y, y):
	return torch.mean(torch.abs(y.sub(y_pred)))


run_id = 'model_gpu{}_n{}_bs{}_lr{}'.format(gpu_id, epochs, batch_size, learning_rate); print('\n\nTraining', run_id)
save_path = run_id + '.pkl'

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

class RMSE(object):
    def __init__(self):
        self.sq_errors = []
        self.num_pix = 0
        
    def get(self):
        return np.sqrt(
                    np.sum(np.array(self.sq_errors))/self.num_pix
                )
    
    def add_batch(self, pred, target):
        sqe = (pred-target)**2
        self.sq_errors.append(np.sum(sqe))
        self.num_pix += target.size
        
    def reset(self):
        self.sq_errors = []
        self.num_pix = 0


# Used to keep track of statistics
class AverageMeter(object):
    def __init__(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

REPORTS_PER_EPOCH = 10
ITER_PER_EPOCH = len(train_loader)
ITER_PER_REPORT = ITER_PER_EPOCH//REPORTS_PER_EPOCH

metrics = RMSE()

for epoch in range(epochs):
    model.train()
    
    # Progress reporting
    batch_time = AverageMeter()
    losses = AverageMeter()
    N = len(train_loader)
    end = time.time()

    for i, (sample) in enumerate(train_loader):

        # Load a batch and send it to GPU
        x = sample['image'].float().cuda()
        y = sample['depth'].float().cuda()

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        
        # Record loss
        losses.update(loss.data.item(), x.size(0))

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model).
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))

        # Log training progress
        if i % ITER_PER_REPORT == 0:
            print('\nEpoch: [{0}][{1}/{2}]\t' 'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t' 'ETA {eta}\t'
             'Training Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))
        elif i % (ITER_PER_REPORT//50) == 0:
            print('.', end='')
            
        #break # useful for quick debugging        
    torch.cuda.empty_cache(); del x, y; gc.collect()
    
    # Validation after each epoch
    model.eval()
    metrics.reset()
    for i, (sample) in enumerate(valid_loader):
        x, y = sample['image'].float().cuda(), sample['depth'].numpy()
        with torch.no_grad():
            y_pred = model(x).detach().cpu().numpy()

        metrics.add_batch(y_pred, y)
        print('_', end='')
    print('\nValidation RMSE {avg_rmse}'.format(avg_rmse=metrics.get()))
    

# Save model
torch.save(model.state_dict(), save_path)
print('\nTraining done. Model saved ({}).'.format(save_path))

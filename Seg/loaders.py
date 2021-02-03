import os
import glob
import torch
import numpy as np
from skimage import io, transform
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split

np.random.seed(0) 
torch.manual_seed(0)
VALIDATION_SPLIT = 0.02

class DepthHalfSize(object):
    def __call__(self, sample):
        x = sample['depth']
        sample['depth'] = transform.resize(x, (x.shape[0]//2, x.shape[1]//2))
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        # swap channel axis
        image = image.transpose((2, 0, 1))
        depth = depth.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'depth': torch.from_numpy(depth)}
    
class DepthToNormal(object):
    def __call__(self, sample):
        dx, dy = np.gradient(sample['depth'].squeeze())
        dx, dy, dz = dx * 2500, dy * 2500, np.ones_like(dy)
        n = np.linalg.norm(np.stack((dy, dx, dz), axis=-1), axis=-1)
        d = np.stack((dy/n, dx/n, dz/n), axis=-1)
        return {'image': sample['image'], 'depth': (d + 1) * 0.5} 
        
class ImageDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform 
        self.image_files = glob.glob(root_dir + '/*.jpg')
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = io.imread(self.image_files[idx]) / 255.0
        depth = io.imread(self.image_files[idx].replace('.jpg', '.png'))[:,:,:1] / 255.0        
        sample = {'image': image, 'depth': depth}        
        return self.transform(sample) if self.transform else sample
    
def prep_loaders(root_dir=None, batch_size=1, workers=1):
    # Load dataset
    image_depth_dataset = ImageDepthDataset(root_dir=root_dir, transform=transforms.Compose([DepthHalfSize(), ToTensor()]))

    # Split into training and validation sets
    train_size = int((1-VALIDATION_SPLIT) * len(image_depth_dataset))
    test_size = len(image_depth_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(image_depth_dataset, [train_size, test_size])

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    print('Dataset size (num. batches)', len(train_loader), len(valid_loader))
    
    return train_loader, valid_loader

##############################################################
################# Segmentation Section #######################
##############################################################

class SegIdentityTransform(object):
    # Hint: Note that our transforms work on dicts. This is an example of a transform that works
    # on a dict whose elements can be converted to np.arrays, and are then converted to torch.tensors
    # This performs the scaling of the RGB by division by 255, and puts channels first by performing the permute
    # for the label, we convert to long, datatype to let torch know that this is a discrete label.
    # You might want to change this or write different transforms depending on how you read data.
    def __call__(self, sample):
        #label_trans = transforms.Compose([transforms.Resize(256)])
        sample['image'] = torch.tensor(np.array(sample['image'])/255.0).permute(2,0,1)
        sample['label'] = torch.tensor(np.array(sample['label'])).long()
        sample['image'] =sample['image'].resize((256, 256))
        tf = transforms.Compose([transforms.Scale(size = 256),transforms.ToTensor()])
        #sample['image'] = tf(sample['image'])
												                                          ])
        #sample['image'] = transform.resize(sample['image'],(256,256),preserve_range=True)
        #sample['label'] = transform.resize(sample['label'],(256,256),preserve_range=True)
        #sample['image'] = transforms.Normalize(sample['image'],[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        #sample['image'] /= 225.0
        #norm = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        #sample['image'] = norm(sample['image'])
        #sample['image'] -= (0.485, 0.456, 0.406)
        #sample['image'] /= (0.485, 0.456, 0.406)

        #return {'image': torch.tensor(np.array(sample['image']/225.0)).permute(2,0,1),
        #        'label': torch.tensor(np.array(sample['label'])).long()}
        return sample

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0],
                       [128, 0, 0],
                       [0, 128, 0],
                       [128, 128, 0],
                       [0, 0, 128],
                       [128, 0, 128],
                       [0, 128, 128],
                       [128, 128, 128],
                       [64, 0, 0],
                       [192, 0, 0],
                       [64, 128, 0],
                       [192, 128, 0],
                       [64, 0, 128],
                       [192, 0, 128],
                       [64, 128, 128],
                       [192, 128, 128],
                       [0, 64, 0],
                       [128, 64, 0],
                       [0, 192, 0],
                       [128, 192, 0],
                       [0, 64, 128]])




def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """

    # TODO (hint: You might not need a lot of work here with some libraries, which already read in the image as a single channel label)
    # (hint: the said library does not return a np.ndarray object)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    if isinstance(mask, np.ndarray):
        # TODO
        mask = mask.astype(int)
        for i, label in enumerate(get_pascal_labels()):
            #print(mask.shape)
            #print(i)
            #print(np.where(np.all(mask == label, axis=-1)))
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
    else:
        # TODO if the image is just single channel
        #  you might want to convert the single channel label to a np.ndarray
        pass
    return label_mask

def decode_segmap(mask, unk_label=255):
    """Decode segmentation label prediction as RGB images
    Args:
        mask (torch.tensor): class map with dimensions (B, M,N), where the value at
        a given location is the integer denoting the class index.
    Returns:
        (np.ndarray): colored image of shape (BM, BN, 3)
    """
    mask[mask == unk_label] == 0
    mask = mask.numpy()
    cmap = get_pascal_labels()
    cmap_exp = cmap[..., None]
    colored = cmap[mask].squeeze()
    grid = make_grid(torch.tensor(colored).permute(0, 3, 1, 2))
    return np.permute(grid, (1, 2, 0))



class VOCSeg(Dataset):
    def __init__(self, root_dir, split=None, transform=None):
        # Known information
        self.num_classes = 21
        self.class_names = ['Background',
                            'Aeroplane',
                            'Bicycle',
                            'Bird',
                            'Boat',
                            'Bottle',
                            'Bus',
                            'Car',
                            'Cat',
                            'Chair',
                            'Cow',
                            'Diningtable',
                            'Dog',
                            'Horse',
                            'Motorbike',
                            'Person',
                            'Pottedplant',
                            'Sheep',
                            'Sofa',
                            'Train',
                            'Tvmonitor']

        # Set up proper paths
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, 'JPEGImages')
        self.label_dir = os.path.join(self.root_dir, 'SegmentationClass')
        
        self.transform = transform 

        #TODO Read the appropriate split file and save the file names
        self.split = split
        self.split_file_dir = os.path.join(self.root_dir, 'ImageSets', 'Segmentation')

        # TODO read in ONLY files from self.split_file
        #self.image_files = None
        #self.label_files = None
        with open(os.path.join(os.path.join(self.split_file_dir, self.split + '.txt')), "r") as f:
            lines = f.read().splitlines()
            self.image_files = [os.path.join(self.root_dir, "JPEGImages", name + ".jpg") for name in lines]
            self.label_files = [os.path.join(self.root_dir, "SegmentationClass", name + ".png") for name in lines]
            print(len(self.image_files))
            
        


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        # TODO Retrieve the saved file names and perform the proper processing
        # The images go from 0-255 to 0-1. You can also use the range -1 to 1
        # The labels go from a 3 channel RGB to a single channel with elements in the range 0..N-1
        #image = None
        image = io.imread(self.image_files[idx])
        #label_rgb = None
        label_rgb = io.imread(self.label_files[idx])
        label = label_rgb[:,:,:3]
        label = encode_segmap(label) # write the encode_segmap function
        sample = {'image': image, 'label': label}
        
        return self.transform(sample)


def get_seg_loaders(root_dir=None, batch_size=1, workers=1):

    #TODO optionally add more augmentation
    tfms = transforms.Compose([
        #transforms.Resize(256),
        #transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]),
        SegIdentityTransform()
    ])

    train_set = VOCSeg(root_dir=root_dir, split='train', transform=tfms)
    val_set = VOCSeg(root_dir=root_dir, split='val', transform=tfms) # No transforms on the validation set

    # Prepare data_loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, valid_loader

if __name__ == '__main__':
    pass

from torchvision.datasets import ImageFolder

from PIL import Image
from torchvision import transforms
import os
import os.path
import torch.nn as nn
import torch.nn.functional as F
from my_transforms import RandomCutImageAndReturnNewClass




IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class JFE_action(ImageFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, shade_cut=False, shade_index = 2):
        super(JFE_action, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.transform = transform
        self.shade_cut = shade_cut
        self.shade_index = shade_index
        self.Totensor = transforms.ToTensor()
        self.cut_trans = RandomCutImageAndReturnNewClass(p=0.08)

    def __getitem__(self, index):

        path, target = self.imgs[index]
        #print(path, target)
        sample = self.loader(path)
        
        #print(sample.size)
        
        if self.transform is not None:
            sample = self.transform(sample)
            #print(sample.shape)
        if self.shade_cut:
            sample, cut = self.cut_trans(sample)
            if cut:
                target = self.shade_index
        h,w = sample.size
        if h>w :
            w =int(w*224/h)
            h =224 
            pad  = nn.ZeroPad2d(padding=(0,0,0,(h-w)))
            sample = sample.resize((h,w))
        else:
            h =int(h*224/w)
            w =224
            pad  = nn.ZeroPad2d(padding=((w-h),0,0,0))
            sample = sample.resize((h,w))
        sample = self.Totensor(sample).float()
        sample = pad(sample)


        return sample, target


    def __len__(self):
        return len(self.samples)

class JFE_action_test(ImageFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(JFE_action_test, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.transform = transforms.Compose([
    transforms.RandomRotation((-15,15), expand = True, center = None),
    transforms.ColorJitter(brightness = (0.8,1.2),contrast = (0.8,2)),
    transforms.RandomHorizontalFlip()
    ])
        self.Totensor = transforms.ToTensor()


    def __getitem__(self, index):

        path, target = self.imgs[index]
        #print(path, target)
        sample = self.loader(path)
        
        #print(sample.size)
        
        if self.transform is not None:
            sample = self.transform(sample)
            #print(sample.shape)
        h,w = sample.size
        if h>w :
            w =int(w*224/h)
            h =224 
            pad  = nn.ZeroPad2d(padding=(0,0,0,(h-w)))
            sample = sample.resize((h,w))
        else:
            h =int(h*224/w)
            w =224
            pad  = nn.ZeroPad2d(padding=((w-h),0,0,0))
            sample = sample.resize((h,w))
        sample = self.Totensor(sample)
        sample = pad(sample)


        return sample, target, path


    def __len__(self):
        return len(self.samples)


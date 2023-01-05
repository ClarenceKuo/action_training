import glob
import cv2
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

def toTensor(img):
    img = np.array(img)
    assert type(img) == np.ndarray
    img = torch.from_numpy(img).permute(2,0,1).float()
    img = img/127.5 - 1
    return img.unsqueeze(0)


def pad_and_resize(img):
      old_shape = img.shape
      ratio = 224/max(old_shape)
      new_shape = [1,3, int(old_shape[2]*ratio), int(old_shape[3]*ratio)]
      #print(old_shape,new_shape)
      img = F.interpolate(img, new_shape[2:])
      h,w = img.shape[2:]
      if w>h :
          pad  = torch.nn.ZeroPad2d(padding=((224-w),0,0,(224-h)))
      else:
          pad  = torch.nn.ZeroPad2d(padding=((224-w),0,0,(224-h)))
      new_img = pad(img)
      #print('out',new_img.shape)
      return new_img.squeeze(0)


class JFE_action(Dataset):
    def __init__(self, img_path, gt_path, transforms=None):
        self.img = img_path
        self.gt = gt_path
        self.preprocess = pad_and_resize
        self.transforms = transforms


    def __getitem__(self, index):
        img, gt = self.img[index], self.gt[index]
        img = cv2.imread(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        img = torch.Tensor(img).unsqueeze(0)
        #img = Image.open(img).convert('RGB')
        if self.transforms is not None:
          img = self.transforms(img)
        img = self.preprocess(img)
        

        return img, gt

    def __len__(self):
        return len(self.img)


def get_path():
    img_path = []
    gt_path = []
    path  = glob.glob('/mnt/demovideo4/JFE_forlabel/JFE_images/bending/*jpg')
    for p in path:
        img_path.append(p)
        gt_path.append(0)

    

    return img_path,gt_path


'''
if __name__=='__main__':

    img, gt = get_path()
    dataset = JFE_action(img,gt)
    train_loader = DataLoader(dataset, batch_size = 1, num_workers = 8, shuffle = True)
    train = iter(train_loader)
    img, gt = next(train)
    print(img.shape, gt)
    '''
import random

import torch
import torch.nn.functional as F

class RandomCutImageAndReturnNewClass(torch.nn.Module):
    def __init__(self, p=0.2, keep_min = 0.2, keep_max = 0.6):
        super().__init__()
        self.p = p
        self.cut_min = keep_min
        self.cut_max = keep_max

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cut.
        Returns:
            PIL Image or Tensor: Randomly cut image.
            bool: Image is cut or not
        """
        cut = False
        if torch.rand(1) < self.p:
            w = 1
            x = 0
            h = random.random()*(self.cut_max - self.cut_min) + self.cut_min
            y = 0 if torch.rand(1) < 0.5 else (1-h)
            x, y, w, h = x*img.size[0], y*img.size[1], w*img.size[0], h*img.size[1]
            img = img.crop((x,y,x+w,y+h))

            cut = True
        return img, cut

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

if __name__ == "__main__":
    from PIL import Image

    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    RC = RandomCutImageAndReturnNewClass(p=1)
    image_path = "/home/ubuntu/Desktop/JFE/XML/actions_4IR/Lying/1-2 (4).0.jpg"
    image = pil_loader(image_path)
    image.save("a.png")
    image_cut, _ = RC(image)
    image_cut.save("b.png")
    print("")
    

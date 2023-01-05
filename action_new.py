import torch
import timm

m = timm.create_model('resnet18', num_classes=3, checkpoint_path='./savemodel/model_best.pth.tar')
o = m(torch.randn(2, 3, 224, 224))
print(f'Pooled shape: {o.shape}')
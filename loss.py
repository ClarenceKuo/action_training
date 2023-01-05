import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class Focal_Loss(_Loss):
    def __init__(self, alpha = 1, gamma = 5, logits = False, reduce = True, ignore_index= 255, from_logits=False):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.logits = logits
        self.reduce = reduce
        #self.weight = torch.Tensor([50.0, 200.0, 1.0]).to(device)
        
    def forward(self, inputs, targets):
        CE = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index)
        pt = torch.exp(-CE)
        F_loss = self.alpha * (1-pt)**self.gamma * CE

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
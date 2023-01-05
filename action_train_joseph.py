import torch 
import time
from optimizer import Ranger
import torch.nn.functional as F
import torchvision
import glob
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn, Tensor
import timm
from dataset import JFE_action, JFE_action_test
#from JFE_dataset import JFE_action
from ghostnet import ghostnet_pretrained,get_action_model_train
from loss import Focal_Loss
from tensorboardX import SummaryWriter
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(39)
valid_size = 0.3
batch_size = 48
num_classes = 6
num_epochs = 150
tfms = transforms.Compose([
    transforms.RandomRotation((-10,10), expand = True, center = None),
    transforms.ColorJitter(brightness = (0.8,1.2),contrast = (0.8,1.2)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomResizedCrop(scale=(0.7,1))
    ])
# img_list =['/mnt/HD/JFE/action/JFE_action/XML/action_1012/output_1014/bending/*.jpg','/mnt/HD/JFE/action/JFE_action/XML/action_1012/output_1014/lying/*.jpg','/mnt/HD/JFE/action/JFE_action/XML/action_1012/output_1014/others/*.jpg','/mnt/HD/JFE/action/JFE_action/XML/action_1012/output_1014/squat/*.jpg','/mnt/HD/JFE/action/JFE_action/XML/action_1012/output_1014/upright/*.jpg']
# img_path = []
# gt_path = []

#def get_path():
#    for i in range(5):
#        path  = glob.glob(img_list[i])
#        for p in path:
#            img_path.append(p)
#            gt_path.append(i)
#
#    return img_path,gt_path

#img_path, gt = get_path()

#path = '/mnt/DpData/training_data/JFE_action/JFE_train1012_J'
path = '/home/ubuntu/Desktop/action_saas/all/'
test_path = '/home/ubuntu/Desktop/action_saas/all/'
model_name = 'action_6_Wresnet50_shadeY_2'
shade_index = 2

img_data = JFE_action(path, transform = tfms, shade_cut=True, shade_index=shade_index)
# img_data_test = JFE_action(test_path, transform = tfms)

# test_data = JFE_action_test(test_path, transform = None )
num_train = len(img_data)
indices = list(range(num_train))
split = int(np.floor(valid_size*num_train))
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_idx=train_idx[:len(train_idx)//batch_size*batch_size]
valid_idx=valid_idx[:len(valid_idx)//batch_size*batch_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(img_data, batch_size = batch_size, sampler = train_sampler, num_workers = 4)
valid_loader = DataLoader(img_data, batch_size = batch_size, sampler = valid_sampler, num_workers = 4)
#test_loader = DataLoader(test_data, batch_size = 1, num_workers = 8, shuffle = True)
#class_counts = dict(Counter(sample_tup[1] for sample_tup in train_loader.dataset))
print(len(train_idx), len(valid_idx))

# model = get_action_model_train(num_classes).to(device)
model = timm.create_model('wide_resnet50_2', pretrained=True, num_classes = num_classes).to(device)
loss = Focal_Loss(gamma = 3)
optimizer = Ranger(model.parameters())
writer = SummaryWriter()
# resume = 'ghostnet_best.pth'
# resume = 'savemodel/action_3G_T_best1.pth'
resume = None
def save_model(state,save_model_path,modelname):
    filename = os.path.join(save_model_path,modelname+'_'+str(state['epoch']+1)+'.pth')
    torch.save(state,filename)

def save_best(state,save_model_path,modelname):
    filename = os.path.join(save_model_path,modelname+'_best'+'.pth')
    torch.save(state,filename)

def load_model(Net, optimizer, model_file):
    assert os.path.exists(model_file),'There is no model file from'+model_file
    checkpoint = torch.load(model_file, map_location='cuda:0')
    Net.load_state_dict(checkpoint)
    start_epoch = 0
    # if optimizer is not None:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return Net, optimizer, start_epoch

def train(epoch, data_loader, model, loss_fn):
    model.train()
    training_loss = 0.0
    correct = 0
    loss = 0
    for batch_idx, (img, gt) in enumerate(data_loader):
        img, gt = img.to(device),gt.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = loss_fn(pred,gt)
        loss.backward()
        optimizer.step()
        training_loss = loss.item()
        pred = pred.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(gt.data.view_as(pred)).cpu().sum().item()
        step = len(data_loader.sampler)//10
    print('Train Epoch: {}, Loss: {:.8f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                epoch, loss.item(), correct, len(train_idx),
            100. * correct / len(train_idx)))
    return training_loss / len(train_idx), 100. * float(correct) / len(train_idx)

def valid(data_loader, model, loss_fn):
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        correct = 0

        for img, gt in data_loader:
            img,gt = img.to(device), gt.to(device)
            pred = model(img)
            valid_loss += loss_fn(pred,gt).item()
            pred = pred.data.max(1, keepdim = True)[1]
            correct += pred.eq(gt.data.view_as(pred)).cpu().sum().item()

        # valid_loss /= len(data_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            valid_loss, correct, len(valid_idx),
            100. * correct / len(valid_idx)))

    return valid_loss, 100*float(correct) / len(valid_idx)

def test(data_loader, model, loss_fn):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        correct = 0

        t1 = time.time()
        for img, gt, path in data_loader:
            img,gt = img.to(device), gt.to(device)
            pred = model(img)
            score = F.softmax(pred, dim = 1)
            #score = score.max(1, keepdim = True)
            confi = score.data.max(1, keepdim = True)[0]
            preds = pred.data.max(1, keepdim = True)[1]
            print(path,confi, preds)
            for t, p in zip(gt.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += preds.eq(gt.data.view_as(preds)).cpu().sum().item()

        t2 = time.time()
        valid_loss /= len(data_loader.dataset)

        print(confusion_matrix)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            valid_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))
    
    fps = len(data_loader.dataset)/(t2-t1)
    print(fps)

    return valid_loss, 100*float(correct) / len(data_loader.dataset)

if resume is not None:
    model, optimizer, start_epoch = load_model(model, optimizer, resume)
    print(resume,'loaded!')

best = 0.8
#test_loss, test_acc = test(test_loader, model, loss)


early_stop = 0
for epoch in range(num_epochs):

    training_loss, train_acc = train(epoch, train_loader, model, loss)
    valid_loss, valid_acc = valid(valid_loader, model, loss)
    writer.add_scalar('loss/train', training_loss, epoch)
    writer.add_scalar('loss/validation', valid_loss, epoch)
    writer.add_scalar('Acc/train', train_acc, epoch)
    writer.add_scalar('Acc/validation', valid_acc, epoch)

    if epoch%10 == 0:
        save_model({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        },
                        'savemodel',model_name)
    elif valid_acc > best and train_acc>= valid_acc:
        # save_best({'epoch':epoch,
        #                 'model_state_dict':model.state_dict(),
        #                 'optimizer_state_dict':optimizer.state_dict(),
        #                 },
        #                 'savemodel',model_name)
        save_best(model.state_dict(), 'savemodel',model_name)
        best = valid_acc
        early_stop = 0
        print('save_best', valid_acc)
    else:
        early_stop += 1
        if early_stop > 50:
            print(f"early stop at epoch: {epoch}")
            break
    
print(f"best: {best}")





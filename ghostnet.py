import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

__all__ = ['ghost_net']



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

    
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.inplanes = output_channel
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)        

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


BN_MOMENTUM = 0.1
class GhostNetCtn(nn.Module):
    def __init__(self, cfgs, heads, head_conv, width=1.0):
        super(GhostNetCtn, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        #self.dropout = dropout
        self.heads = heads
        self.head_conv = head_conv
        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel
        self.deconv_with_bias = False

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)
        self.inplanes = 960

        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )
        # self.final_layer = []

        for head in sorted(self.heads):
          num_output = self.heads[head]
          if head_conv > 0:
            fc = nn.Sequential(
                nn.Conv2d(256, head_conv,
                  kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output, 
                  kernel_size=1, stride=1, padding=0))
          else:
            fc = nn.Conv2d(
              in_channels=256,
              out_channels=num_output,
              kernel_size=1,
              stride=1,
              padding=0
          )
          self.__setattr__(head, fc)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)

        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNet(cfgs, **kwargs)

def load_weight(weight, model):
    weight_name = torch.load(weight, map_location=lambda storage, loc:storage)
    dict_new = model.state_dict().copy()
    #print(dict_new.keys())
    new_list = list(model.state_dict().keys())
    train_list = list(weight_name['model_state_dict'].keys())
    #print(train_list)
    key = []
    for x in train_list:
        if x in new_list:
            #print('in', x)
            key.append(x)
        if x not in new_list:
            print(x)
    for tensor_name in key:
        print(tensor_name)
        dict_new[tensor_name]=weight_name['model_state_dict'][tensor_name]
    model.load_state_dict(dict_new)

def load_pretrained_weight(weight, model):
    weight_name = torch.load(weight, map_location=lambda storage, loc:storage)
    dict_new = model.state_dict().copy()
    #print(dict_new.keys())
    new_list = list(model.state_dict().keys())
    train_list = list(weight_name.keys())
    #print(train_list)
    key = []
    for x in train_list:
        if x in new_list:
            #print('in', x)
            key.append(x)
        if x not in new_list:
            print(x)
    for tensor_name in key:
        print(tensor_name)
        dict_new[tensor_name]=weight_name[tensor_name]
    model.load_state_dict(dict_new) 		

def ghostnet_pretrained(classes):
    model = ghostnet()
    weight = 'state_dict_93.98.pth'
    #model.load_state_dict(weight)
    load_pretrained_weight(weight,model)
    model.classifier = nn.Linear(1280,classes,True)

    return model

def get_GhostnetCnt(heads, head_conv):
    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNetCtn(cfgs, heads, head_conv)

def get_action_model():
    model = ghostnet()
    model.classifier = nn.Linear(1280, 5, True)
    weight = '/home/ubuntu/Desktop/ghostnet_JFE_best.pth'
    load_weight(weight,model)

    return model

def get_action_model_train(class_num):
    model = ghostnet()
    model.classifier = nn.Linear(1280, class_num, True)

    return model

def toTensor(img):
    import numpy as np
    img = np.array(img, dtype=np.float16)
    # print(type(img), img.shape, img.max(), img.min())
    img = img/127.5 - 1.0
    assert type(img) == np.ndarray
    img = torch.from_numpy(img).permute(2,0,1).float()
    # img = img/127.5 - 1.0
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
      return new_img


if __name__=='__main__':
    import time
    from torch import nn
    import torch.nn.functional as F
    from PIL import Image
    import glob
    from torchvision import transforms
    import os
    import numpy as np
    #imgs = glob.glob('/mnt/program_code/Olivia_goods/CenterNet/src/test_img/*.jpg')
    imgs = glob.glob('/mnt/demovideo4/JFE_forlabel/JFE_images/bending/*jpg')
    model = get_action_model()
    model = model.cuda()
    
    Totensor = transforms.ToTensor()
    with torch.no_grad():
        model.eval()
        for i,img in enumerate(imgs):
            img_ = Image.open(img).convert('RGB')
            img_in = img_.copy.deepcopy()
           
            img_in = toTensor(img_in).float()
            #img_in = Totensor(img_).float()
            #print(img_in.shape)
            img_in = pad_and_resize(img_in).cuda()
            # img_in = img_.cuda()
            #img_out = (img_in+1)*127.5
            #img_show = transforms.ToPILImage()(img_in.cpu().squeeze(0)).convert('RGB')

            #np_img = np.array(img_show)

            #print("np_img.max: {} , np_img.min : {}".format( np_img.max(), np_img.min()))
            #img_show.show()
            pred = model(img_in)
            action = pred.data.max(1, keepdim=True)[1]
            #cv2.imshow('result',img_)
            print(img,pred, action, type(pred))

            if i == 5:
                break

    


'''
    heads = {'hm': 5, 'dep': 1, 'rot': 8, 'dim': 3}
    head_conv = 256
    model = get_GhostnetCnt(heads, head_conv)
    weight = 'state_dict_93.98.pth'
    load_weight(weight,model)
    #model.classifier = nn.Linear(1280,5,True)
 
    model = model.cuda()
    dummy_input = torch.randn(1,3,224,224).cuda()
    out = model(dummy_input)
    print(out)
    #input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    #output_names = [ "output1" ]

    #torch.onnx.export(model, dummy_input, "ghostnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

    with torch.no_grad():
        t1 = time.time()
        y = model(input)
        t2 = time.time()
        #print(y.size())
        #print(1/(t2-t1))
'''
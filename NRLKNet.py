# The code is adapted from VanillaNet and RepVGG.
# Original repository: [VanillaNet]: https://github.com/huawei-noah/VanillaNet, 
#                      [RepVGG]: https://github.com/DingXiaoH/RepVGG

import torch 
import torch.nn as nn
import numpy as np


class activation(nn.ReLU):
    def __init__(self, dim, act_num=(1,3), deploy=False):
        super(activation, self).__init__()
        self.act_num = act_num
        self.deploy = deploy
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num[0]*2 + 1, act_num[1]*2 + 1))
        if deploy:
            self.bias = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None
            self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x), 
                self.weight, self.bias, padding=self.act_num, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=self.act_num, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,25),
                 stride=1, padding='same', dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(Block, self).__init__()
        self.deploy = deploy
        self.act_learn = 1
        self.in_channels = in_channels

        self.act = activation(out_channels, act_num=(1,1), deploy=self.deploy)
        self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)) 

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)

            self.sub_1x1 = conv_bn(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=groups)

            self.cross_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.cross_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)

    def forward(self, inputs):
        if self.deploy:
            return self.pool(self.act(self.rbr_reparam(inputs)))    

        if self.rbr_identity is None:   
            id_out = 0
            cr_out = 0
        else:
            id_out = self.rbr_identity(inputs)
            cr_out = self.cross_identity(inputs)
        
        x = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        x = self.sub_1x1(x) 
        x1 = self.cross_1x1(inputs) + cr_out
        return self.pool(self.act(x + x1))
    
    def get_equivalent_kernel_bias(self):
        w_rbr_dense, b_rbr_dense = self._fuse_bn_tensor(self.rbr_dense)
        _,_,h,w = w_rbr_dense.shape
        w_rbr_1x1, b_rbr_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        w_rbr_1x1 = self._pad_1x1_tensor(w_rbr_1x1, (h,w))
        w_rbr_id, b_rbr_id = self._fuse_bn_tensor(self.rbr_identity)
        if isinstance(w_rbr_id, torch.Tensor): 
            w_rbr_id = self._pad_1x1_tensor(w_rbr_id, (h,w))
        w1, b1 = w_rbr_dense + w_rbr_1x1 + w_rbr_id, b_rbr_dense + b_rbr_1x1 + b_rbr_id
        
        # 合并sub_1x1 和 bn
        w_sub_1x1, b_sub_1x1 = self._fuse_bn_tensor(self.sub_1x1)
        
        # 合并rbr_dense 和 sub_1x1
        w2 = torch.einsum('oi,icjk->ocjk', w_sub_1x1.squeeze().squeeze(), w1)
        b2 = (b1.view(1,-1,1,1)*w_sub_1x1).sum(3).sum(2).sum(1) + b_sub_1x1
        
        # 合并最外层的cross连接
        w_cross_1x1, b_cross_1x1 = self._fuse_bn_tensor(self.cross_1x1)
        w_cross_id, b_cross_id = self._fuse_bn_tensor(self.cross_identity)
        if isinstance(w_cross_id, torch.Tensor): 
            w_cross_id = self._pad_1x1_tensor(w_cross_id, (h,w))
        return w2 + self._pad_1x1_tensor(w_cross_1x1, (h,w)) + w_cross_id, b2 + b_cross_1x1 + b_cross_id

    def _pad_1x1_tensor(self, kernel1x1, size):
        if kernel1x1 is None:
            return 0
        else:
            pad_h, pad_w = size[0] // 2, size[1] // 2
            return torch.nn.functional.pad(kernel1x1, [pad_w,pad_w,pad_h,pad_h])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):    # fuse 3x3conv-bn & 1x1conv-bn
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)    # 仅BN
            
            kernel_value = np.zeros((self.in_channels, self.in_channels, 1, 1), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        self.__delattr__('sub_1x1')
        self.__delattr__('cross_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'cross_identity'):
            self.__delattr__('cross_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.act.switch_to_deploy()
        self.deploy = True
        

class NRLKNet(nn.Module):
    def __init__(self, num_classes, in_channels=1, dims=[88, 48, 48, 80], kernel_sizes=[(3,25),(1,25),(1,25),(1,25)], drop_rate=0, deploy=False):
        super().__init__()

        self.in_channels = in_channels
        self.dims = dims
        self.deploy = deploy
        
        self.stages = nn.ModuleList()
        for i in range(len(dims)):
            stage = Block(self.in_channels, dims[i], kernel_size=kernel_sizes[i], padding='same', deploy=deploy)
            self.in_channels = dims[i]
            self.stages.append(stage)
        self.depth = len(dims)
        
        if self.deploy:
            self.cls = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Dropout(drop_rate),
                nn.Conv2d(dims[-1], num_classes, 1),
            )
        else:
            self.cls1 = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Dropout(drop_rate),
                    nn.Conv2d(dims[-1], num_classes, 1),
                    nn.BatchNorm2d(num_classes),
                )
            self.cls2 = nn.Sequential(
                nn.Conv2d(num_classes, num_classes, 1)
            )

    def forward(self, x):
        for i in range(self.depth):
            x = self.stages[i](x)

        if self.deploy:
            x = self.cls(x)
        else:
            x = self.cls1(x)
            x = self.cls2(x)

        return x.view(x.size(0),-1)
    
    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        for i in range(self.depth):
            self.stages[i].switch_to_deploy()

        kernel, bias = self._fuse_bn_tensor(self.cls1[2], self.cls1[3])
        self.cls1[2].weight.data = kernel
        self.cls1[2].bias.data = bias
        kernel, bias = self.cls2[0].weight.data, self.cls2[0].bias.data
        self.cls1[2].weight.data = torch.matmul(kernel.transpose(1,3), self.cls1[2].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.cls1[2].bias.data = bias + (self.cls1[2].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.cls = torch.nn.Sequential(*self.cls1[0:3])
        self.__delattr__('cls1')
        self.__delattr__('cls2')
        self.deploy = True
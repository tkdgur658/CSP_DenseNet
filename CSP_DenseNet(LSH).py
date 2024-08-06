import torch
import torch.nn as nn
import numpy as np
def divide_2_as_int(number):
    return int(np.round(number/2))

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
class TransitionLayer(nn.Module):
    def __init__(self, inplace, plance):
        super(TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace,out_channels=plance,kernel_size=1,stride=1,padding=0,bias=False),
            nn.AvgPool2d(kernel_size=2,stride=2),
        )

    def forward(self, x):
        return self.transition_layer(x)
class CSP_TransitionLayer(nn.Module):
    def __init__(self, inplace, plance):
        super(CSP_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace,out_channels=plance,kernel_size=1,stride=1,padding=0,bias=False),
        )

    def forward(self, x):
        return self.transition_layer(x)

class _DenseLayer(nn.Module):
    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, inplances, growth_rate, bn_size , drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
   
class CSP_Transition(nn.Sequential):
    def __init__(self, inplace, plance):
        super(CSP_Transition, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace,out_channels=plance,kernel_size=1,stride=1,padding=0,bias=False),
        )
    def forward(self, x):
        return self.transition_layer(x)  

class CSP_Block(torch.nn.Module):
    def __init__(self, num_layers, inplances, bn_size, growth_rate, drop_rate, Transition=True):
        super(CSP_Block,self).__init__()
        self.Transition = Transition
        self.partial_index = divide_2_as_int(inplances)
        inplances = divide_2_as_int(inplances)
        dense_block = DenseBlock(num_layers, inplances, growth_rate, bn_size , drop_rate=0)#(torch.rand((1,16,56,56))).shape
        inplances = inplances + num_layers * growth_rate
        
        
        small_transition = CSP_TransitionLayer(inplances, divide_2_as_int(inplances))
        inplances = divide_2_as_int(inplances)
        self.right_layers =nn.Sequential(
            dense_block, 
            small_transition
        )
        if inplances%2 == 1:
            inplances = inplances + self.partial_index-1
        else:
            inplances = inplances + self.partial_index
        if self.Transition == True:
            self.big_trainsition = TransitionLayer(inplances, divide_2_as_int(inplances)) 
            inplances = divide_2_as_int(inplances)
        self.out_channels = inplances
    def return_out_channels(self):
        return self.out_channels
    def forward(self,x):
        x_through = x[:,:self.partial_index]
        x_direct= x[:,self.partial_index:]
        x_through = self.right_layers(x_through)
        x = torch.cat([x_direct,x_through],1)
        if self.Transition == True:
            x = self.big_trainsition(x)
        return x

        
class CSP_DenseNet(nn.Module):
    def __init__(self, in_channels, num_classes=1000, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16]):
        super(CSP_DenseNet, self).__init__()
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(in_planes=in_channels, places=init_channels)

        num_features = init_channels
        
        self.layer1 = CSP_Block(num_layers=blocks[0], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = self.layer1.return_out_channels()
        
        self.layer2 = CSP_Block(num_layers=blocks[1], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = self.layer2.return_out_channels()
        
        self.layer3 = CSP_Block(num_layers=blocks[2], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = self.layer3.return_out_channels()
        
        self.layer4 = CSP_Block(num_layers=blocks[3], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate, Transition=False)
        num_features = self.layer4.return_out_channels()
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
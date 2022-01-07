
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class Bottleneck(nn.Module):
    def __init__(self, channels, in_channels=-1):
        super(Bottleneck, self).__init__()
        self.BNmomentum = 0.9
        self.channels = channels
        if (in_channels == -1):
            self.in_channels = channels
            self.skipLayer = nn.Identity()
        else:
            self.in_channels = in_channels
            self.skipLayer = nn.Conv2d(in_channels,channels,1,1)
        self.mainConv = nn.Sequential(
            nn.BatchNorm2d(self.in_channels, self.BNmomentum),
            nn.ReLU(inplace = True),
            nn.Conv2d(self.in_channels, channels//2, 1, 1, 0),
            nn.BatchNorm2d(channels//2, self.BNmomentum),
            nn.ReLU(inplace = True),
            nn.Conv2d(channels//2, channels//2, 3, 1, 1),
            nn.BatchNorm2d(channels//2, self.BNmomentum),
            nn.ReLU(inplace = True),
            nn.Conv2d(channels//2, channels, 1, 1, 0)
        )
        self.relu = nn.ReLU(inplace = True)
    def forward(self, X):
        conv = self.mainConv(X)
        X = self.skipLayer(X)
        return torch.add(conv, X)
        
class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
    def forward(self, X):
        return self.conv(X)

class HourGlass(nn.Module):
    def __init__(self, order, channels, num_residual):
        super(HourGlass, self).__init__()
        self.order = order
        self.channels = channels
        self.num_residual = num_residual
        self.upper_bottleneck = nn.Sequential(*[Bottleneck(channels) for i in range(self.num_residual)])
        self.max_pool = nn.MaxPool2d(2,2)
        self.lower_bottleneck = nn.Sequential(*[Bottleneck(channels) for i in range(self.num_residual)])
        self.output_bottleneck = nn.Sequential(*[Bottleneck(channels) for i in range(self.num_residual)])
        if order > 1:
            self.hg = HourGlass(order-1, channels, num_residual)
        else:
            self.deep_residuals = nn.Sequential(*[Bottleneck(channels) for i in range(self.num_residual)])
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, X):
        #Upper branch
        up1 = X
        up1 = self.upper_bottleneck(up1)
        
        #Lower branch
        low1 = self.max_pool(X)
        low1 = self.lower_bottleneck(low1)
        
        low2 = low1
        if self.order > 1:
            #Go into deeper level of hourglass
            low2 = self.hg(low2)
        else:
            #Achieved deepest level, perform residuals at the latent space
            low2 = self.deep_residuals(low2)
        
        low3 = low2
        low3 = self.output_bottleneck(low3)
        up2 = self.upsample(low3)
        if up2.shape != up1.shape:  
            up2 = TF.resize(up2, size=up1.shape[2:])
        return torch.add(up2,up1)

class BeginStackedHourGlass(torch.nn.Module):
    def __init__(self, in_channels,out_channels):
        super(BeginStackedHourGlass, self).__init__()
        self.begin = nn.Sequential(
            nn.Conv2d(in_channels,64,7,2,3),
            nn.BatchNorm2d(momentum=0.9,num_features=64),
            nn.ReLU(inplace=True),
            Bottleneck(128, in_channels=64),
            nn.MaxPool2d(2,2),
            Bottleneck(128),
            Bottleneck(out_channels, in_channels = 128)
        )
    def forward(self, X):
        return self.begin(X)

class StackedHourGlass(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=21, num_stacks = 1, num_residual=1, hourglass_depth=4,n_features=256):
        super(StackedHourGlass, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.relu = nn.ReLU(inplace=True)
        self.num_stacks = num_stacks
        self.num_residual = num_residual
        self.begin = BeginStackedHourGlass(in_channels,n_features)
        self.hourglasses = nn.ModuleList([HourGlass(hourglass_depth,n_features,self.num_residual) for i in range(self.num_stacks)])
        self.intermediate_bottlenecks = nn.ModuleList([nn.Sequential(*[Bottleneck(n_features) for j in range(self.num_residual)]) for i in range(self.num_stacks)])
        self.linear_layers = nn.ModuleList([LinearLayer(n_features,n_features) for i in range(self.num_stacks)])
        self.Ys = nn.ModuleList([nn.Conv2d(n_features,self.out_channels,1) for i in range(self.num_stacks)])
        self.intermediate1s = nn.ModuleList([nn.Conv2d(n_features,n_features,1) for i in range(self.num_stacks-1)])
        self.intermediate2s = nn.ModuleList([nn.Conv2d(self.out_channels,n_features,1) for i in range(self.num_stacks-1)])
    def forward(self, X):
        X = self.begin(X)
        ys = []
        for i in range(self.num_stacks):
            X = self.hourglasses[i](X)

            X = self.intermediate_bottlenecks[i](X)
            X = self.linear_layers[i](X)

            Y = self.Ys[i](X)
            ys.append(Y)
            if i < self.num_stacks -1:
                y_intermediate_1 = self.intermediate1s[i](X)
                y_intermediate_2 = self.intermediate2s[i](Y)
                X = torch.add(y_intermediate_1,y_intermediate_2)
        return ys
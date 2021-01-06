import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url

# model
class DMVSTNet(nn.Module):
    def __init__(self, spatial_hidden_size):
        super().__init__()
        self.spatial_hidden_size = spatial_hidden_size
        self.H = 10
        self.W = 20
        self.n_seq = 1

        # spatial module
        spatial_layer1 = nn.Sequential()
        spatial_layer1.add_module('conv1', nn.Conv2d(2, 32, kernel_size=(3,3), padding=(1,1)))
        spatial_layer1.add_module('activation1', nn.ReLU())
        #spatial_layer1.add_module('dropout1', nn.Dropout(p=0.25))
        spatial_layer1.add_module('batchnorm1', nn.BatchNorm2d(32))
        self.spatial_layer1 = spatial_layer1

        spatial_layer2 = nn.Sequential()
        spatial_layer2.add_module('conv2', nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1)))
        spatial_layer2.add_module('activation2', nn.ReLU())
        #spatial_layer2.add_module('dropout2', nn.Dropout(p=0.25))
        spatial_layer2.add_module('batchnorm2', nn.BatchNorm2d(32))
        self.spatial_layer2 = spatial_layer2

        spatial_layer3 = nn.Sequential()
        spatial_layer3.add_module('conv3', nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1)))
        spatial_layer3.add_module('activation3', nn.ReLU())
        self.spatial_layer3 = spatial_layer3

        spatial_layer4 = nn.Sequential()
        spatial_layer4.add_module('fc4', nn.Linear(in_features=32*self.H*self.W, out_features=200))
        self.spatial_layer4 = spatial_layer4


    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # (seg_len, batch_size, n_feature, H, W)
        # spatial part
        x = self.spatial_layer1(x)
        x = self.spatial_layer2(x)
        x = self.spatial_layer3(x)
        x = torch.flatten(x, start_dim = 1)
        output = self.spatial_layer4(x)

        return output


import torch
import torch.nn as nn

# model
class DMVSTNet(nn.Module):
    def __init__(self, input_size, temporal_hidden_size):
        super().__init__()
        self.temporal_hidden_size = temporal_hidden_size
        self.H = 10
        self.W = 20
        self.n_seq = 7

        # spatial module
        # spatial_layer1 = nn.Sequential()
        # spatial_layer1.add_module('conv1', nn.Conv2d(2, self.spatial_hidden_size, kernel_size=(3,3), padding=(1,1)))
        # spatial_layer1.add_module('activation1', nn.ReLU())
        # #spatial_layer1.add_module('dropout1', nn.Dropout(p=0.25))
        # spatial_layer1.add_module('batchnorm1', nn.BatchNorm2d(self.spatial_hidden_size))
        # self.spatial_layer1 = spatial_layer1

        # spatial_layer2 = nn.Sequential()
        # spatial_layer2.add_module('conv2', nn.Conv2d(self.spatial_hidden_size, self.spatial_hidden_size, kernel_size=(3,3), padding=(1,1)))
        # spatial_layer2.add_module('activation2', nn.ReLU())
        # #spatial_layer2.add_module('dropout2', nn.Dropout(p=0.25))
        # spatial_layer2.add_module('batchnorm2', nn.BatchNorm2d(self.spatial_hidden_size))
        # self.spatial_layer2 = spatial_layer2

        # spatial_layer3 = nn.Sequential()
        # spatial_layer3.add_module('conv3', nn.Conv2d(self.spatial_hidden_size, self.spatial_hidden_size, kernel_size=(3,3), padding=(1,1)))
        # spatial_layer3.add_module('activation3', nn.ReLU())
        # self.spatial_layer3 = spatial_layer3

        # spatial_layer4 = nn.Sequential()
        # spatial_layer4.add_module('fc4', nn.Linear(in_features=self.spatial_hidden_size*self.H*self.W, out_features=64))
        # spatial_layer4.add_module('activation4', nn.ReLU())
        # self.spatial_layer4 = spatial_layer4

        # temporal module
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=temporal_hidden_size,
        )
        self.fc = nn.Linear(self.temporal_hidden_size, 200)



    def forward(self, x, h_lstm):
        x = x.permute(1, 0, 4, 2, 3) # (batch_size, seg_len, n_feature, H, W)
        x = torch.flatten(x, start_dim=2)
        output, h_t = self.lstm(x)
        output = self.fc(output[0])

        return output


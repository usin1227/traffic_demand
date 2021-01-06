import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, n_head, out_size):
        super(Transformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nhead = n_head
        self.out_size = out_size

        # self.pos_encoder = PositionalEncoding(input_size)
        self.Wh = nn.Linear(input_size, hidden_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=n_head)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=6)
        # self.decoder = nn.Linear(hidden_size, out_size) # single step prediction for all zone\
        self.fc = nn.Linear(hidden_size, out_size)

    
    def forward(self,src):
        # src = self.pos_encoder(src)
        input = self.Wh(src)
        output = self.transformer_encoder(input)
        output = self.fc(output)
        return output[-1]


# model
class DMVSTNet(nn.Module):
    def __init__(self, spatial_hidden_size, temporal_hidden_size):
        super().__init__()
        self.spatial_hidden_size = spatial_hidden_size
        self.temporal_hidden_size = temporal_hidden_size
        self.H = 10
        self.W = 20
        self.n_seq = 7
        self.n_head = 1
        

        # spatial module
        spatial_layer1 = nn.Sequential()
        spatial_layer1.add_module('conv1', nn.Conv2d(2, 32, kernel_size=(3,3), padding=(1,1)))
        spatial_layer1.add_module('activation1', nn.ReLU())
        spatial_layer1.add_module('dropout1', nn.Dropout(p=0.25))
        spatial_layer1.add_module('batchnorm1', nn.BatchNorm2d(32))
        self.spatial_layer1 = spatial_layer1

        spatial_layer2 = nn.Sequential()
        spatial_layer2.add_module('conv2', nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1)))
        spatial_layer2.add_module('activation2', nn.ReLU())
        spatial_layer2.add_module('dropout2', nn.Dropout(p=0.25))
        spatial_layer2.add_module('batchnorm2', nn.BatchNorm2d(32))
        self.spatial_layer2 = spatial_layer2

        spatial_layer3 = nn.Sequential()
        spatial_layer3.add_module('conv3', nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1)))
        spatial_layer3.add_module('activation3', nn.ReLU())
        self.spatial_layer3 = spatial_layer3

        spatial_layer4 = nn.Sequential()
        spatial_layer4.add_module('fc4', nn.Linear(in_features=32*self.H*self.W, out_features=64))
        spatial_layer4.add_module('activation4', nn.ReLU())
        self.spatial_layer4 = spatial_layer4

        
        # # temporal module
        # self.lstm = nn.LSTM(self.spatial_hidden_size, self.temporal_hidden_size)
        self.trans = Transformer(64, self.temporal_hidden_size, self.n_head, self.H*self.W)
        
        # final fc module
        # fc_layer = nn.Sequential()
        # fc_layer.add_module('fc_final1', nn.Linear(in_features=self.temporal_hidden_size, out_features=self.H*self.W))
        # fc_layer.add_module('activation_final1', nn.Sigmoid())
        # self.fc_layer = fc_layer


    def forward(self, x, h_s, h_lstm):
        x = x.permute(1, 0, 4, 2, 3) # (seg_len, batch_size, n_feature, H, W)

        # spatial part
        for seq in range(len(x)):
            h1 = self.spatial_layer1(x[seq])
            h1 = self.spatial_layer2(h1)
            h1 = self.spatial_layer3(h1)
            h_s[seq] = h1
        h_s = torch.flatten(h_s, start_dim=2)
        h_s = self.spatial_layer4(h_s)

        # temporal part
        # out, h_t = self.lstm(h_s, h_lstm)
        output = self.trans(h_s)
        #out, h_t = self.temporal_layer1(h_s) #h_lstm
        # output = self.fc_layer(h_t[0])

        return output


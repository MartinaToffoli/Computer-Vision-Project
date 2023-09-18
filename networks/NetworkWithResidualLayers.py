import torch.nn as nn

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
    super(Block, self).__init__()
    self.expansion = 2
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size = 1, stride = 1, padding = 0)
    self.bn3= nn.BatchNorm2d(out_channels * self.expansion)
    self.relu = nn.ReLU()
    self.identity_downsample = identity_downsample

  def forward(self, x):
    identity = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)

    if self.identity_downsample is not None:
        identity = self.identity_downsample(identity)

    x += identity
    x = self.relu(x)
    return x

class ResidualNetwork(nn.Module):

  def __init__(self, Block, layers, in_channels, num_classes=26):
    
    super(ResidualNetwork, self).__init__()
    # input size should be : (b x 1 x 28 x 28)
    self.in_channels = 50
    self.conv1 = nn.Conv2d(in_channels=1 , out_channels=50, kernel_size=(3, 3), stride=(1,1), padding=(1,1))
    self.bn1 = nn.BatchNorm2d(50)
    self.relu1 = nn.ReLU()
    self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding = 1)   
    
    self.layer1 = self.make_layer(Block, layers[0], out_channels=75, stride=1)
    self.layer2 = self.make_layer(Block, layers[1], out_channels=100, stride=2) 
    
    self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding = 1)  
    
    self.fc = nn.Sequential(
      nn.Dropout(p=0.5, inplace=True),
      nn.Linear(in_features=200 * 4 * 4, out_features=4096),
      nn.ReLU(),
      nn.Dropout(p=0.5, inplace=True),
      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(),
      nn.Linear(in_features=4096, out_features=num_classes),
    )

  def forward(self, x):
   
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.maxpool1(x)
    
    x = self.layer1(x)
    x = self.layer2(x)
    
    x = self.maxpool2(x)
    
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)

    return x
  
  def make_layer(self, Block, num_residual_blocks, out_channels, stride):
    identity_downsample = None
    layers = []


    if stride != 1 or self.in_channels != out_channels * 2:
        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * 2, kernel_size = 1, stride = stride),
                                            nn.BatchNorm2d(out_channels * 2))
    layers.append(Block(self.in_channels, out_channels, identity_downsample, stride))
    self.in_channels = out_channels * 2

    for i in range(num_residual_blocks - 1):
        layers.append(Block(self.in_channels, out_channels))

    return nn.Sequential(*layers)

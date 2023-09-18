import torch.nn as nn

class LeNet_modified(nn.Module):
  def __init__(self, num_classes=26):
    super().__init__()
    # input size should be : (b x 1 x 28 x 28)
    self.net = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=2), 
      nn.ReLU(),
      nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k=2),  
      nn.MaxPool2d(kernel_size=3, stride=2),  
      
      nn.Conv2d(6, 24, 3, padding=0),  
      nn.ReLU(),
      nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=3, stride=2),  
    )
    # classifier is just a name for linear layers
    self.classifier = nn.Sequential(
      nn.Dropout(p=0.5, inplace=True),
      nn.Linear(in_features=(24 * 5 * 5), out_features=120),
      nn.ReLU(),
      nn.Dropout(p=0.5, inplace=True),
      nn.Linear(in_features=120, out_features=84),
      nn.ReLU(),
      nn.Linear(in_features=84, out_features=num_classes),
    )

  def forward(self, x):
    x = self.net(x)
    x = x.view(-1, 24 * 5 * 5)  # reduce the dimensions for linear layer input
    return self.classifier(x)

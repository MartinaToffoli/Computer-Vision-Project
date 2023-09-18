import torch.nn as nn

class Network_28x28(nn.Module):
  def __init__(self, num_classes=26):
    super().__init__()
    # input size should be : (b x 1 x 28 x 28)
    self.net = nn.Sequential(
      
      nn.Conv2d(in_channels=1, out_channels=50, kernel_size=3, stride=1),  
      nn.ReLU(),
      nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=3, stride=2),  
      
      nn.Conv2d(50, 150, 3, padding=2),  
      nn.ReLU(),
      nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=3, stride=2),
      
      nn.Conv2d(150, 200, 3, padding=1), 
      nn.ReLU(),
      nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=3, stride=2),      
    )
    
    self.classifier = nn.Sequential(
      
      nn.Dropout(p=0.5, inplace=True),
      nn.Linear(in_features=(200 * 2 * 2), out_features=4096),
      nn.ReLU(),
      
      nn.Dropout(p=0.5, inplace=True),
      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(),
      
      nn.Linear(in_features=4096, out_features=num_classes),

    )
  
  def forward(self, x):
    x = self.net(x)
    x = x.view(-1, 200 * 2 * 2)
    return self.classifier(x)

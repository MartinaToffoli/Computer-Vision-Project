import torch.nn as nn

class Network_100x100(nn.Module):
  def __init__(self, num_classes=26):
    super().__init__()
    # input size should be : (b x 1 x 100 x 100)
    self.net = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(),
      nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
      
      nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(),
      nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
      
      nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
      
      nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(),
      
      nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
    )

    self.classifier = nn.Sequential(
      nn.Dropout(p=0.5, inplace=True),
      nn.Linear(in_features=(256 * 5 * 5), out_features=4096),
      nn.ReLU(),
      nn.Dropout(p=0.5, inplace=True),
      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(),
      nn.Linear(in_features=4096, out_features=num_classes),
    )

  def forward(self, x):
    x = self.net(x)
    x = x.view(-1, 256 * 5 * 5)
    return self.classifier(x)

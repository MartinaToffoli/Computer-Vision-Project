import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from itertools import cycle
from networks import Network_28x28_mmd
from utilities import csv_loader
from utilities import check_accuracy

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (for the network 28 x 28)
in_channels = 1
num_classes = 26
learning_rate = 1e-3  # Adam: 1e-5
batch_size = 40  # 60
num_epochs = 60


# Load data

dataset_test = csv_loader.MyDataSet('csv_img_unito.csv')
test_loader_homemade = data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
print('* Test dataset homemade Dataset reading...done')


# Load model

model = Network_28x28_mmd.Network_28x28().to(device)

#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate , momentum=0.9)

checkpoint = torch.load("model.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Test 

check_accuracy.check_accuracy(test_loader_homemade, model, -1, 'Homemade Dataset')

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from itertools import cycle
from networks import Network_28x28
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
dataset_train = csv_loader.MyDataSet('sign_mnist_train.csv')
train_loader = data.DataLoader(dataset_train, batch_size=batch_size)
print('* Train dataset reading...done')


dataset_test = csv_loader.MyDataSet('sign_mnist_test.csv')
test_loader = data.DataLoader(dataset_test, batch_size=batch_size)
print('* Test dataset reading...done')

# Initialize model
model = Network_28x28.Network_28x28().to(device)

#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate , momentum=0.9)

accuracy_test_old = 0
last_epoch = 0
accuracy = 0

# Training network
for epoch in range(last_epoch, num_epochs):

    for data_sup in train_loader:   
        data_model_sup = torch.reshape(data_sup[0][:,0,:,:], (batch_size, 1, 28, 28))  
        
        # Get data to cuda if possible
        data_model_sup = data_model_sup.to(device=device)
        targets = data_sup[1].to(device=device)

        # forward
        scores_sup = model(data_model_sup)

        loss = F.cross_entropy(scores_sup, targets)
             
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trainAcc = check_accuracy.check_accuracy(train_loader, model, epoch + 1)
    testAcc = check_accuracy.check_accuracy(test_loader, model, -1, 'MNIST Test')
    #testUnito = check_accuracy.check_accuracy(test_loader_homemade, model, -1, 'Homemade Dataset')

    # Save best model
    accuracy = float(testAcc)
    
    if accuracy > accuracy_test_old:
      accuracy_test_old = accuracy
      torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            },"model"+str(epoch+1)+".pth")
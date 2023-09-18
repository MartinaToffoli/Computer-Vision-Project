import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_accuracy(loader, model, batch_size, epoch = -1, name='Train'):
  num_correct = 0
  num_samples = 0
  model.eval()

  with torch.no_grad():
    for x, y in loader:
      if x.shape[0] == batch_size:
        x = torch.reshape(x[:,0,:,:], (batch_size, 1, 28, 28)) 
        x = x.to(device)
        y = y.to(device)

        scores = model(x)  # Forward
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum().item()
        num_samples += predictions.size(0)

    if epoch != -1:
      print("--------")
      print(f'Training --> Epoch n. [{epoch}] - Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    else:
      print(f'Test {name} --> Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

  model.train()

  return "{:.2f}".format(float(num_correct) / float(num_samples) * 100)
 
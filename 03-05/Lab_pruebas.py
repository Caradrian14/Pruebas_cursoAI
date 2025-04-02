import torch
import numpy as np
import matplotlib.pyplot as plt
from fontTools.unicodedata import block
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

input_dim = 1
output_dim = 1
learning_rate = 0.1

# This is our weight matrix
w = torch.tensor([100.0],requires_grad=True,dtype=torch.float32)
# This is our bias vector
b = torch.zeros(size=(output_dim,),requires_grad=True)
n = 100
X, Y = make_classification(n_samples = n, n_features=2,
                           n_redundant=0, n_informative=2, flip_y=0.1,class_sep=1.5)
X = X.astype(np.float32)
Y = Y.astype(np.int32)
split = [ 70*n//100, (15+70)*n//100 ]
train_x, valid_x, test_x = np.split(X, split)
train_labels, valid_labels, test_labels = np.split(Y, split)
dataset = torch.utils.data.TensorDataset(torch.tensor(train_x),torch.tensor(train_labels,dtype=torch.float32))
dataloader = torch.utils.data.DataLoader(dataset,batch_size=16)
val_x = torch.tensor(valid_x)
val_lab = torch.tensor(valid_labels)
#-----------------------------

class MyNet(torch.nn.Module):
  def __init__(self, hidden_size=10, func=torch.nn.Sigmoid()):
    super().__init__()
    self.fc1 = torch.nn.Linear(2, hidden_size)
    self.func = func
    self.fc2 = torch.nn.Linear(hidden_size, 1)

  def forward(self, x):
    x = self.fc1(x)
    x = self.func(x)
    x = self.fc2(x)
    return x


net = MyNet(func=torch.nn.ReLU())
print(net)
def train(net, dataloader, val_x, val_lab, epochs=10, lr=0.05):
  optim = torch.optim.Adam(net.parameters(),lr=lr)
  for ep in range(epochs):
    for (x,y) in dataloader:
      z = net(x).flatten()
      loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
      optim.zero_grad()
      loss.backward()
      optim.step()
    acc = ((torch.sigmoid(net(val_x).flatten())>0.5).float()==val_lab).float().mean()
    print(f"Epoch {ep}: last batch loss = {loss}, val acc = {acc}")

train(net,dataloader,val_x,val_lab,lr=0.005)


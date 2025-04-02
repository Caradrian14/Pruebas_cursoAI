import torch
import numpy as np
import matplotlib.pyplot as plt
from fontTools.unicodedata import block
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import random
import pytorch_lightning as pl
import matplotlib.pyplot as plt
print(torch.cuda.is_available())
exit()
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


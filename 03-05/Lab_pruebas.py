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

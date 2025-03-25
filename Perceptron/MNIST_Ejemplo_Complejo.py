# Ejemplo para resolver problemas complejos NO PODEMOS CORRERLO POR QUE NO HAY DATOS DE IMAGENES QUE PUEDAN SERVIR
# PERO PONEMOS EL CODIGO IGUALMENTE POR QUE MOLA

# If you are not running this notebook from a cloned repository, you may need to grab the binary dataset file first
# !wget https://github.com/microsoft/AI-For-Beginners/blob/main/data/mnist.pkl.gz?raw=true
# In this case correct the link to the dataset below as well.
import pylab
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import pickle
import os
import gzip

# no hay, no existe
with gzip.open('../../data/mnist.pkl.gz', 'rb') as mnist_pickle:
    MNIST = pickle.load(mnist_pickle)

print(MNIST['Train']['Features'][0][130:180])
print(MNIST['Train']['Labels'][0])
features = MNIST['Train']['Features'].astype(np.float32) / 256.0
labels = MNIST['Train']['Labels']
fig = pylab.figure(figsize=(10,5))
for i in range(10):
    ax = fig.add_subplot(1,10,i+1)
    pylab.imshow(features[i].reshape(28,28))
pylab.show()


def set_mnist_pos_neg(positive_label, negative_label):
    positive_indices = [i for i, j in enumerate(MNIST['Train']['Labels'])
                        if j == positive_label]
    negative_indices = [i for i, j in enumerate(MNIST['Train']['Labels'])
                        if j == negative_label]

    positive_images = MNIST['Train']['Features'][positive_indices]
    negative_images = MNIST['Train']['Features'][negative_indices]

    fig = pylab.figure()
    ax = fig.add_subplot(1, 2, 1)
    pylab.imshow(positive_images[0].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(1, 2, 2)
    pylab.imshow(negative_images[0].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    pylab.show()

    return positive_images, negative_images

pos1,neg1 = set_mnist_pos_neg(1,0)



def plotit2(snapshots_mn,step):
    fig = pylab.figure(figsize=(10,4))
    ax = fig.add_subplot(1, 2, 1)
    pylab.imshow(snapshots_mn[step][0].reshape(28, 28), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    pylab.colorbar()
    ax = fig.add_subplot(1, 2, 2)
    ax.set_ylim([0,1])
    pylab.plot(np.arange(len(snapshots_mn[:,1])), snapshots_mn[:,1])
    pylab.plot(step, snapshots_mn[step,1], "bo")
    pylab.show()
def pl3(step): plotit2(snapshots_mn,step)
def pl4(step): plotit2(snapshots_mn2,step)

snapshots_mn = train_graph(pos1,neg1,1000)
interact(pl3, step=widgets.IntSlider(value=0, min=0, max=len(snapshots_mn) - 1))

#For some reason, 2 and 5 are not as easily separable. Even though we get relatively high accuracy (above 85%), we can clearly see how perceptron stops learning at some point.
# Esto se debe a una limitacion del machine lerning, a la hora de hacer los graficos los puntos caen en muchas coincidencias, y puede llevar a errores. Ademas que son similares en diseño el 2 y 5

from sklearn.decomposition import PCA


def pca_analysis(positive_label, negative_label):
    positive_images, negative_images = set_mnist_pos_neg(positive_label, negative_label)
    M = np.append(positive_images, negative_images, 0)

    mypca = PCA(n_components=2)
    mypca.fit(M)

    pos_points = mypca.transform(positive_images[:200])
    neg_points = mypca.transform(negative_images[:200])

    pylab.plot(pos_points[:, 0], pos_points[:, 1], 'bo')
    pylab.plot(neg_points[:, 0], neg_points[:, 1], 'ro')

pca_analysis(1, 0)

pca_analysis(2,5)
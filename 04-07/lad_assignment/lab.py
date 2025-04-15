# Classification of Pet's Faces

# aparentemente han quitado el enlace a las imagenes
# !wget https://mslearntensorflowlp.blob.core.windows.net/data/petfaces.tar.gz



import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def display_images(l,titles=None,fontsize=12):
    n=len(l)
    fig,ax = plt.subplots(1,n)
    for i,im in enumerate(l):
        ax[i].imshow(im)
        ax[i].axis('off')
        if titles is not None:
            ax[i].set_title(titles[i],fontsize=fontsize)
    fig.set_size_inches(fig.get_size_inches()*n)
    plt.tight_layout()
    plt.show()

# Now let's traverse all class subdirectories and plot first few images of each class:

for cls in os.listdir('petfaces'):
    print(cls)
    display_images([Image.open(os.path.join('petfaces',cls,x))
                    for x in os.listdir(os.path.join('petfaces',cls))[:10]])



# Let's also define the number of classes in our dataset:
num_classes = len(os.listdir('petfaces'))
num_classes

# ------------- start CODE TO LOAD DATASET -------------

# ------------- end CODE TO LOAD DATASET -------------


# -------------start CODE TO DO TRAIN/TEST SPLIT -------------

# ------------- end CODE TO DO TRAIN/TEST SPLIT -------------

# ------------- Print tensor sizes -------------

# ------------- Display the data -------------

# ------------- CODE TO DEFINE NEURAL NETWORK

# TRAIN THE NETWORK

# PLOT THE ACCURACY on train and validation dataset

# Define dataset that contains only two labels: 0 = cat, 1 = dog
# Hint: use class name prefix to figure out which one is which
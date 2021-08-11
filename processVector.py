from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inputV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_inputVGG16
from keras.utils import plot_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import skimage.feature as feature
import matplotlib.pyplot as plt

import numpy as np

print("Reading file...")
data = np.loadtxt("inceptionList.dat")

print(data)

tsne = TSNE(n_components=2, random_state=0)
intermediates_tsne = tsne.fit_transform(data)
plt.scatter(x = intermediates_tsne[:,0], y=intermediates_tsne[:,1], color=color_intermediates)
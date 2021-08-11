from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import time
import os

pca = PCA(n_components=2)

print("Loading simple elements array...")
reduced_data_simple = np.loadtxt('reduced_features.txt', dtype=np.float64)
print("Array loaded")
print(reduced_data_simple)
print(str(len(reduced_data_simple)))

reduced_data_simple = pca.fit_transform(reduced_data_simple)

plt.scatter(reduced_data_simple[:, 0], reduced_data_simple[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title('Distribución de datos')
plt.savefig('raw_data.jpg')
print(pca.explained_variance_ratio_)
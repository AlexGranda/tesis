from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inputV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_inputVGG16
from keras.utils import plot_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import skimage.feature as feature
import matplotlib.pyplot as plt

import numpy as np
import os

rootdir ='/Users/laagranda/Downloads/Tesis/Documents/ProcessedFaces'
landscapesRootDir = '/Users/laagranda/Downloads/Tesis/Documents/LandscapeImages'

modelInceptionV3 = InceptionV3(weights='imagenet', include_top=False)
modelInceptionV3.summary()
inceptionV3_feature_list = []

modelVGG16 = VGG16(weights='imagenet', include_top=False)
modelVGG16.summary()
VGG16_feature_list = []

listOfFiles = list()

listOfLandscapes = list()
for (dirpath, dirnames, filenames) in os.walk(rootdir):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

for (dirpath, dirnames, filenames) in os.walk(landscapesRootDir):
    listOfLandscapes += [os.path.join(dirpath, file) for file in filenames]

#Comparing between inceptionV3 and VGG16
# Print the files
for elem in listOfFiles:
     print(elem)

for file in listOfLandscapes:
    if file.endswith(".jpg"):
        img = image.load_img(file, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_inputV3(img_data)
        inceptionv3_feature = modelInceptionV3.predict(img_data)

        inceptionV3_feature_np = np.array(inceptionv3_feature)
        inceptionV3_feature_list.append(inceptionV3_feature_np.flatten())

        #With VGG16
        img_data = preprocess_inputVGG16(img_data)
        VGG16_feature = modelInceptionV3.predict(img_data)

        VGG16_feature_np = np.array(VGG16_feature)
        VGG16_feature_list.append(inceptionV3_feature_np.flatten())
        print("Added "+file+ "to array")

print("Processing faces...")

for file in listOfFiles:
    if file.endswith(".jpg") or file.endswith(".png"):
        #With InceptionV3
        img = image.load_img(file, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_inputV3(img_data)
        inceptionv3_feature = modelInceptionV3.predict(img_data)

        inceptionV3_feature_np = np.array(inceptionv3_feature)
        inceptionV3_feature_list.append(inceptionV3_feature_np.flatten())

        #With VGG16
        img_data = preprocess_inputVGG16(img_data)
        VGG16_feature = modelInceptionV3.predict(img_data)

        VGG16_feature_np = np.array(VGG16_feature)
        VGG16_feature_list.append(inceptionV3_feature_np.flatten())
        print("Added "+file+ "to array")



inceptionV3_feature_list_np = np.array(inceptionV3_feature_list)
np.savetxt('inceptionList.dat', inceptionV3_feature_list_np)
'''kmeans = KMeans(n_clusters=2, random_state=0).fit(inceptionV3_feature_list_np)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

plt.scatter(inceptionV3_feature_list_np[:, -1], inceptionV3_feature_list_np[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black', marker='*', s=150)
plt.title('With InceptionV3')
plt.savefig('InceptionV3.jpg')'''

VGG16_feature_list_np = np.array(VGG16_feature_list)
np.savetxt('VGGList.dat', VGG16_feature_list_np)
'''kmeans = KMeans(n_clusters=2, random_state=0).fit(VGG16_feature_list_np)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

plt.scatter(VGG16_feature_list_np[:, -1], VGG16_feature_list_np[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black', marker='*', s=150)
plt.title('With VGG16')
plt.savefig('VGG16.jpg')'''



""" img_path = '/Users/laagranda/Downloads/Tesis/ProcessedFaces/YaleProcessedFaces/TesisProcessedFacesB19/0_0.png'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data) """

print 'Feature vector'

print inceptionv3_feature.shape

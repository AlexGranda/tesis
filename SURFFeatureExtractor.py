import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print('OpenCV Version (should be 3.1.0, with nonfree packages installed, for this tutorial):')
print(cv2.__version__)
print(cv2.__file__)

rootdir ='/Users/laagranda/Downloads/Tesis/Documents/ProcessedFaces'
landscapesRootDir = '/Users/laagranda/Downloads/Tesis/Documents/LandscapeImages'

surf = cv2.xfeatures2d.SURF_create(400)
surf.setHessianThreshold(500)

listOfFiles = list()

listOfLandscapes = list()
images = []

for (dirpath, dirnames, filenames) in os.walk(rootdir):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

for (dirpath, dirnames, filenames) in os.walk(landscapesRootDir):
    listOfLandscapes += [os.path.join(dirpath, file) for file in filenames]

print("Read all paths...")

for file in listOfLandscapes:
    if file.endswith(".jpg"):
        try:
                image = cv2.imread(file)
                kp, desc = surf.detectAndCompute(image,None)
                desc_as_array = np.array(desc, object)
                print("Appending images to list")
                images.append(desc_as_array.flatten())
                print("Added "+file+" to the array")    
        except Exception:
                print("Couldn't read image "+file)
                continue
        

for file in listOfFiles:
    if file.endswith(".jpg") or file.endswith(".png"):
        try:
                image = cv2.imread(file)
                kp, desc = surf.detectAndCompute(image,None)
                desc_as_array = np.array(desc, object)
                images.append(desc_as_array.flatten())
                print("Added "+file+" to the array")
        except Exception:
                continue

print("Raw vectors")
print(images)

images_np = np.array(images, object)
reduced_data = PCA(n_components=2).fit_transform(images_np)
kmeans = KMeans(n_clusters=2, random_state=0).fit(reduced_data)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black', marker='*', s=150)
plt.title('With SIFT')
plt.savefig('SIFT.jpg')
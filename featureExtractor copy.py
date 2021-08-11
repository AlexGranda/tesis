from keras import backend as K
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inputV3
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg16 import preprocess_input as preprocess_inputVGG16
import sklearn
from keras.utils import plot_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
import skimage.feature as feature
import matplotlib.pyplot as plt
import time
                
import numpy as np
import os

rootdir ='/home/ec2-user/Documents/ProcessedFaces'
landscapesRootDir = '/home/ec2-user/Documents/LandscapeImages'

modelInceptionV3 = InceptionV3(weights='imagenet', include_top=False)
modelInceptionV3.summary()
inceptionV3_feature_list = []

"""modelVGG16 = VGG16(weights='imagenet', include_top=False)
modelVGG16.summary()
VGG16_feature_list = []"""

listOfFiles = list()

listOfLandscapes = list()
for (dirpath, dirnames, filenames) in os.walk(rootdir):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

for (dirpath, dirnames, filenames) in os.walk(landscapesRootDir):
    listOfLandscapes += [os.path.join(dirpath, file) for file in filenames]
              
print("Number of faces: "+str(len(listOfFiles)))
print("Number of landscapes: "+str(len(listOfLandscapes)))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
time.sleep(10)

#Comparing between inceptionV3 and VGG16
# Print the files    
for elem in listOfFiles:
     print(elem)   

for file in listOfLandscapes:
    if file.endswith(".jpg"):
        try:
                img = image.load_img(file, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                #With InceptionV3
                img_data = preprocess_inputV3(img_data)
                inceptionv3_feature = modelInceptionV3.predict(img_data)

                inceptionV3_feature_np = np.array(inceptionv3_feature)
                inceptionV3_feature_list.append(inceptionV3_feature_np.flatten())
                print("Added "+file+" to array")
        except Exception:
                print("Couldn't open file: "+file)
                continue

        #With VGG16
        """img_data = preprocess_inputVGG16(img_data)
        VGG16_feature = modelVGG16.predict(img_data)

        VGG16_feature_np = np.array(VGG16_feature)
        VGG16_feature_list.append(VGG16_feature_np.flatten())
        print("Added "+file+" to array")"""

print("Processing faces...")

for file in listOfFiles:
    if file.endswith(".jpg") or file.endswith(".png"):
        img = image.load_img(file, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        #With InceptionV3
        img_data = preprocess_inputV3(img_data)
        inceptionv3_feature = modelInceptionV3.predict(img_data)

        inceptionV3_feature_np = np.array(inceptionv3_feature)
        inceptionV3_feature_list.append(inceptionV3_feature_np.flatten())
        print("Added "+file+" to array")

        #With VGG16
        """img_data = preprocess_inputVGG16(img_data)
        VGG16_feature = modelVGG16.predict(img_data)

        VGG16_feature_np = np.array(VGG16_feature)
        VGG16_feature_list.append(VGG16_feature_np.flatten())"""
        

inceptionV3_feature_list_np = np.array(inceptionV3_feature_list)
K.clear_session()
faces = np.random.choice([0,0], size=26123)
landscapes = np.random.choice([1,1], size=16877)
result = np.concatenate((faces, landscapes), axis=None)
#np.savetxt('inceptionList.dat',inceptionV3_feature_list_np)
n = inceptionV3_feature_list_np.shape[0] # how many rows we have in the dataset
print("Number of rows in the dataset: "+str(n))
chunk_size = 1000 # how many rows we feed to IPCA at a time, the divisor of n
#reduced_data = IncrementalPCA(n_componentr=2, batch_size=10).fit_transform(inceptionV3_feature_list_np)
ipca = IncrementalPCA(n_components=100, batch_size=100)
reduced_data = None
print("N//chunck_size: "+str(n//chunk_size))

for i in range(0, 42):
    print("Fitting pca until "+ str((i+1))*chunk_size)
    if i == 0:
        reduced_data = ipca.fit_transform(inceptionV3_feature_list_np[i*chunk_size : (i+1)*chunk_size])
    elif i != 42:
        reduced_data = np.vstack((reduced_data, ipca.fit_transform(inceptionV3_feature_list_np[i*chunk_size : (i+1)*chunk_size])))
    else:
        continue
    print("Reduced data until now: ")
    print(reduced_data)
    print(str(len(reduced_data)))

print("Executing KMeans...")
print("Reduced data before kMeans:"+str(len(reduced_data)))
#wcss = []
#for i in range (1,16): #15 cluster
#    kmeans = KMeans(n_clusters = i, init='k-means++', random_state=0) 
#    kmeans.fit(reduced_data)
#    wcss.append(kmeans.inertia_)
np.savetxt('reduced_features.txt', reduced_data, fmt='%d')
kmeans = KMeans(n_clusters=4, random_state=0, n_jobs=1, init='k-means++').fit(reduced_data)
#plt.plot(range(1,16),wcss)
#plt.title('Elbow Method')
#plt.xlabel('Number of clusters')
##plt.ylabel('wcss')
#plt.savefig("Elbow.jpg")
print(kmeans.cluster_centers_)
print(kmeans.labels_)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black', marker='*', s=150)
print("Plotting...")
plt.title('With InceptionV3')
plt.savefig('InceptionV3.jpg')
print("Finished plotting")
print(f"Reduced data size:"+str(len(reduced_data)))

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(reduced_data, np.array(result), test_size=0.25, random_state=42)
model = KNeighborsClassifier(n_neighbors=6, n_jobs=3)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

"""VGG16_feature_list_np = np.array(VGG16_feature_list)
reduced_data = PCA(n_components=2).fit_transform(VGG16_feature_list_np)
kmeans = KMeans(n_clusters=2, random_state=0).fit(reduced_data)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black', marker='*', s=150)
plt.title('With VGG16')
plt.savefig('VGG16.jpg')

img_path = '/Users/laagranda/Downloads/Tesis/ProcessedFaces/YaleProcessedFaces/TesisProcessedFacesB19/0_0.png'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

#print('Feature vector')

#print(inceptionv3_feature.shape)

"""

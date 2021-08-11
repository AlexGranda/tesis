from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inputV3
from keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import time
import os

rootdir ='/Users/laagranda/Downloads/Tesis/Documents/ProcessedFaces1'
landscapesRootDir = '/Users/laagranda/Downloads/Tesis/Documents/LandscapeImages1'

modelInceptionV3 = InceptionV3(weights='imagenet', include_top=False)
modelInceptionV3.summary()
inceptionV3_feature_list = []

listOfFiles = list()

listOfLandscapes = list()
for (dirpath, dirnames, filenames) in os.walk(rootdir):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

for (dirpath, dirnames, filenames) in os.walk(landscapesRootDir):
    listOfLandscapes += [os.path.join(dirpath, file) for file in filenames]

print("Number of faces: "+str(len(listOfFiles)))
print("Number of landscapes: "+str(len(listOfLandscapes)))
print("Shape of the reduced_data array")
time.sleep(10)

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

inceptionV3_feature_list_np = np.array(inceptionV3_feature_list)

reduced_test_data = PCA(n_components=100).fit_transform(inceptionV3_feature_list_np)
np.savetxt('test_features.txt', reduced_test_data, fmt='%.6f')

kmeans = KMeans(n_clusters=4, random_state=0, n_jobs=1, init='k-means++').fit_transform(reduced_test_data)
np.savetxt('test_features_after_KMeans.txt', kmeans, fmt='%.6f')
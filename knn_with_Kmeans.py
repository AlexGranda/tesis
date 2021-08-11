from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import time
import os

print("Loading array...")
reduced_data = np.loadtxt('reduced_features_after_KMeans.txt', dtype = np.float64)
print("Array loaded")
print(reduced_data)
print(str(len(reduced_data)))

test_faces = np.random.choice([0,0], size=1803)
test_landscapes = np.random.choice([1,1], size=1727)
test_results = np.concatenate((test_landscapes, test_faces), axis=None)

kmeans = np.loadtxt('test_features_after_KMeans.txt', dtype=np.float64)

faces = np.random.choice([0,0], size=18123)
landscapes = np.random.choice([1,1], size=16877)
result = np.concatenate((landscapes, faces), axis=None)

print("Result...")
print(result)
print("Size of result: "+str(len(result)))

print("Training kNN...")
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(reduced_data, np.array(result), test_size=0.25, random_state=42)
model = KNeighborsClassifier(n_neighbors=7, n_jobs=3)
model.fit(trainFeat, trainLabels)

acc = model.score(testFeat, testLabels)
print("Accuracy: "+str(acc))
#cm = confusion_matrix(y_test, y_pred)
#print("Confusion matrix, without normalization")
#print(cm)

print("Shape of the training data")
print(trainFeat.shape)
print("Shape of the testing data")
print(kmeans.shape)

print("Test")
Test_predict = model.predict(kmeans)
acc = accuracy_score(test_results, Test_predict)
print("Test Accuracy: "+str(acc))

cm = confusion_matrix(test_results, Test_predict)

cmap = plt.cm.Blues
classes = np.array([0,1])
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="kNN using KMeans Confusion matrix",
           ylabel='True label', xlabel='Predicted label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.savefig("kNN_CM_KMeans.jpg")

print("Training error")
training_predictions = model.predict(trainFeat)
train_error = mean_absolute_error(training_predictions, trainLabels)
print("Training error: "+str(train_error))

print("Testing error")
test_error = mean_absolute_error(Test_predict, test_results)
print("Testing error: "+str(test_error))
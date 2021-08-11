from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np
import time
import os
print("Loading array...")
reduced_data = np.loadtxt('reduced_features_after_KMeans.txt', dtype=np.float64)
print("Array loaded")
print(reduced_data)
print(str(len(reduced_data)))

test_faces = np.random.choice([0,0], size=1727)
test_landscapes = np.random.choice([1,1], size=1803)
test_results = np.concatenate((test_landscapes, test_faces), axis=None)

kmeans = np.loadtxt('test_features_after_KMeans.txt', dtype=np.float64)
print("Test features")
print(kmeans)

faces = np.random.choice([0,0], size=18123)
landscapes = np.random.choice([1,1], size=16877)
result = np.concatenate((landscapes, faces), axis=None)

print("Result...")
print(result)
print("Size of result: "+str(len(result)))

print("Rescaling...")
ss = StandardScaler()
ss.fit(reduced_data)
reduced_data = ss.transform(reduced_data)
ss.fit(kmeans)
kmeans = ss.transform(kmeans)

print("Training...")
X_train, X_test, y_train, y_test = train_test_split(reduced_data, np.array(result), test_size=0.25, random_state=42)
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
cmap = plt.cm.Blues
classes = np.array([0,1])
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Linear Regression Confusion matrix using KMeans",
           ylabel='True label', xlabel='Predicted label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.savefig("Linear_regression__Kmeans_CM.jpg")

print('Model accuracy is: ', accuracy)

print("Confusion matrix, without normalization")
print(cm)

print("Testing...")
test_predictions = LR.predict(kmeans)
test_accuracy = accuracy_score(test_predictions, test_results)
print('Testing accuracy is: ', test_accuracy)

print("Training error")
training_predictions = LR.predict(X_train)
train_error = mean_absolute_error(training_predictions, y_train)
print("Training error: "+str(train_error))

print("Testing error")
test_error = mean_absolute_error(test_predictions, test_results)
print("Testing error: "+str(test_error))
import sys
import ast
import numpy as np
from sklearn import model_selection, svm
from time import time

file_in = "utteranceFeatures/utteranceFeatures.txt"

f = open(file_in,'r')

lines = f.readlines()
utteranceFeatures = []
labels = []
for i in lines:
	utteranceFeatures.append(ast.literal_eval(i)[0])
	labels.append(ast.literal_eval(i)[1])

for i in range(len(labels)):
	labels[i] = np.argmax(labels[i])

start_time = time()

X_train, X_test, y_train, y_test = model_selection.train_test_split(utteranceFeatures, labels, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)
end_time = time()
score = clf.score(X_test, y_test)


print(score, end_time - start_time)
import sys
import numpy as np
from sklearn import model_selection, svm
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from time import time

file_in = sys.argv[1]

# rows_max = 1000

f = open(file_in,'r')

lines = f.readlines()
utteranceFeatures = []
labels = []
for i in lines:
	utteranceFeatures.append(ast.literal_eval(i)[0])
	labels.append(ast.literal_eval(i)[1])

start_time = time()

X_train, X_test, y_train, y_test = model_selection.train_test_split(utteranceFeatures, labels, test_size=0.2, random_state=int(sys.argv[2]))

# clf1 = MultinomialNB()
# clf1.fit(X_train, y_train)
# end_time = time()
# score1 = clf1.score(X_test, y_test)


clf2 = svm.SVC(C=float(sys.argv[3]))
clf2.fit(X_train, y_train)
end_time = time()
score2 = clf2.score(X_test, y_test)


print(score2, end_time - start_time)
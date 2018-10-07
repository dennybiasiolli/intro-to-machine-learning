import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
pred = clf.predict([[-0.8, -1]])
print('pred', pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, np.array([1]))
print('acc', acc)

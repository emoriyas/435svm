import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from random import uniform
from sklearn.datasets import make_blobs

#http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py

def generate_dataset(n, x_bounds, y_bounds, margin):
    """
    Generate dataset with constraints for separation line y=x
    """
    dataset = []

    for __ in range(n):
        x = uniform(*x_bounds)
        y = uniform(*y_bounds)
        distance = abs(x-y)/2**0.5

        while distance < margin:
            x = uniform(*x_bounds)
            y = uniform(*y_bounds)
            distance = abs(x-y)/2**0.5

        dataset.append([[x, y], 1 if y > x else -1])

    return dataset


# figure number
fignum = 1

# fit the model
ker = "linear"
training_size = 20
start = -10
end = 10
margin = 0.0

training_set = generate_dataset(training_size, [start, end], [start, end], margin)
temp = generate_dataset(training_size, [start, end], [start, end], margin)
X = np.zeros(shape=(training_size - 1,2))
Y = np.zeros(shape=(training_size - 1))

for i in range(0, training_size - 1):
	print(temp[i][0])
	print(np.array(temp[i]))
	print("----")
	#X[i] = np.array(temp[i])
	X[i] = temp[i][0]
	Y[i] = temp[i][1]
#X = np.array(X)
#X = np.random.random((training_size, 2))
#Y = [0] * 50 +  [1] *  50

#X, Y = make_blobs(n_samples=training_size, centers=2, random_state = 6)

print(Y)

#clf = svm.SVC(kernel=ker, gamma=2)
clf = svm.SVC(kernel=ker, C=1000)
clf.fit(X, Y)

# plot the line, the points, and the nearest vectors to the plane
plt.figure(fignum, figsize=(4, 3))
plt.clf()

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
            facecolors='none', zorder=10, edgecolors='k')
plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
            edgecolors='k')

plt.axis('tight')
x_min = -10
x_max = 10
y_min = -10
y_max = 10

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.figure(fignum, figsize=(4, 3))
#plt.gca()

plt.plot(0, y_min, 0, y_max)

plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], alpha=0.5,
            levels=[-.5, 0, .5])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())
fignum = fignum + 1
plt.show()

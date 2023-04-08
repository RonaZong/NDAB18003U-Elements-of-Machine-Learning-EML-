import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

# Exercise 1
def read_data(filename):
    '''Check read_data function'''
    input = open(filename, 'r')
    lines = input.read().splitlines()
    newData = []
    for line in lines[1:]:
        line = line.split()
        newData.append([float(line[1]), float(line[2])])
    return newData
def transpose_data(list):
    '''Check transpose_data function'''
    list1 = []
    list2 = []
    for tup in list:
        list1.append(tup[0])
        list2.append(tup[1])
    return list1, list2
data = read_data('faithful.csv')
data = transpose_data(data)
data = np.array(data)
xdata = data[0] #eruptions
ydata = data[1] #waitings
print("Number of training instances: %i" % data.shape[1])
print("Number of test instances: %i" % data.shape[1])
print("Number of features: %i" % data.shape[0])

plt.plot(xdata, ydata, 'o', color='green')
plt.title("Data")
plt.savefig("Data")
plt.show()

x = np.array(xdata).reshape((xdata.shape[0], -1))
y = np.array(ydata).reshape((ydata.shape[0], -1))

X_train = np.hstack([x, y])

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
clf.fit(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(0, 6)
y = np.linspace(0, 10)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)
plt.contour(X, Y, Z)
plt.scatter(xdata, ydata, 5, c='green')
plt.title("Density Estimation")
plt.savefig("Density Estimation")
plt.show()

# Exercise 2
def generate_X():
    X = X_train
    plt.scatter(X[:, 0], X[:, 1], s=5)
    plt.show()
    return X

def update_W(X, Mu, Var, Pi):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    return W

def update_Pi(W):
    Pi = W.sum(axis=0) / W.sum()
    return Pi

def logLH(X, Pi, Mu, Var):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    return np.mean(np.log(pdfs.sum(axis=1)))

def plot_clusters(X, Mu, Var, Mu_true=None, Var_true=None):
    colors = ['b', 'g', 'r']
    n_clusters = len(Mu)
    plt.scatter(X[:, 0], X[:, 1], s=5)
    ax = plt.gca()
    for i in range(n_clusters):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
        ellipse = Ellipse(Mu[i], Var[i][0], Var[i][1], **plot_args)
        ax.add_patch(ellipse)
    plt.savefig("EM")
    plt.show()

def update_Mu(X, W):
    n_clusters = W.shape[1]
    Mu = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Mu[i] = np.average(X, axis=0, weights=W[:, i])
    return Mu

def update_Var(X, Mu, W):
    n_clusters = W.shape[1]
    Var = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Var[i] = np.average((X - Mu[i]) ** 2, axis=0, weights=W[:, i])
    return Var

eruptions_mean = np.mean(xdata)
waiting_mean = np.mean(ydata)
print(eruptions_mean)
print(waiting_mean)
eruptions_var = np.var(xdata)
waiting_var = np.var(ydata)
print(eruptions_var)
print(waiting_var)

X = generate_X()
n_clusters = 1
n_points = len(X)
Mu = [[eruptions_mean, waiting_mean]]
Var = [[eruptions_var, waiting_var]]
Pi = [1 / n_clusters]
W = np.ones((n_points, n_clusters)) / n_clusters
Pi = W.sum(axis=0) / W.sum()
loglh = []
for i in range(5):
    plot_clusters(X, Mu, Var)
    loglh.append(logLH(X, Pi, Mu, Var))
    W = update_W(X, Mu, Var, Pi)
    Pi = update_Pi(W)
    Mu = update_Mu(X, W)
    print('log-likehood:%.3f'%loglh[-1])
    Var = update_Var(X, Mu, W)

# Exercise 3

# Exercise 4
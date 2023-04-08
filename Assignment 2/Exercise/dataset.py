import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split

def load_data(Train=False):
    data = np.loadtxt("spambase.data", delimiter=",")

    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.float)
    # Free up the memory
    del data

    if Train:
        # Returns X_train, X_test, y_train, y_test
        return train_test_split(X, y, test_size=0.3, random_state=RandomState())
    else:
        return X, y
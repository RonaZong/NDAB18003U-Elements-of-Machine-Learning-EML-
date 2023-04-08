from dataset import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import operator

word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference"]

iterations = 50
k = 3

def find_hyperparams(clf, X, y):
    # Set the parameters by cross-validation
    param_grid = [{'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]}]
    grid = GridSearchCV(clf, param_grid)
    grid.fit(X, y)
    print('done fitting')
    return grid.best_estimator_

def top_k_features(k, weights):
    return sorted(zip(word_labels, weights), reverse=True, key=operator.itemgetter(1))[:k]

def show_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

scores = []
roc_auc = []

for i in range(iterations):
    X_train, X_test, y_train, y_test = load_data(Train=True)

    X_train = X_train[:, 0:48]
    X_test = X_test[:, 0:48]

    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    roc_auc.append(auc(fpr, tpr))

show_auc(y_test, clf.predict_proba(X_test)[:, 1])
print(top_k_features(k, clf.feature_importances_))
print('Accuracy. Avg: %0.5f, Std: %0.5f' % (np.mean(scores), np.std(scores)))
print('AUC. Avg: %0.5f, Std: %0.5f' % (np.mean(roc_auc), np.std(roc_auc)))

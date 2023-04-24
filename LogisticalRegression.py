import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():
    def predict(X_train, y_train, X_test):
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred
    
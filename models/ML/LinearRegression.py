from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, f1_score

def fit_printResult(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    confusion_matrix(y_test, pred)
    plt.show()

def main():
    pass
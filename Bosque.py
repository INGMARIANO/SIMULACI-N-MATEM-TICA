from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics; from time import sleep

def RegresionBosque(X, y, arboles, profundidad) :
    clf = RandomForestClassifier(n_estimators=arboles, max_depth=profundidad); X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, random_state=10); sleep(2); clf.fit(X_train, y_train); pred = clf.predict(X_test)
    return clf, metrics.f1_score(y_test, pred, average="micro")
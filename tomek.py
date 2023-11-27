import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

file_path = 'australian.dat'
data = np.loadtxt(file_path)
#cechy i etykiety
X, y = data[:, :-1], data[:, -1]
#dane testowe i treningowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#klasyfikatory
ada_boost = AdaBoostClassifier()
gradient_boosting = GradientBoostingClassifier()
random_forest = RandomForestClassifier()
svc = SVC(probability=True) #obliczanie prawdopodobienstwa
logistic_regression = LogisticRegression()
knn = KNeighborsClassifier()

#klasyfikatory
classifiers = [
    ada_boost,
    gradient_boosting,
    random_forest,
    svc,
    logistic_regression,
    knn
]
#nazwy klasyfikatorow
classifier_names = [
    'AdaBoost',
    'GradientBoosting',
    'RandomForest',
    'SVC',
    'LogisticRegression',
    'KNN'
]

#slownik efektywnosci
effectiveness = {}

# trenowanie klasyfikatorow i liczenie efektywnosci
for clf, name in zip(classifiers, classifier_names):
    # normalizacja danych
    pipeline = make_pipeline(StandardScaler(), clf)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    effectiveness[name] = accuracy_score(y_test, predictions)

#efektywnosc kumulatywna
cumulative_scores = np.cumsum(list(effectiveness.values()))
cumulative_effectiveness = cumulative_scores / np.arange(1, len(effectiveness) + 1)

#wykres
plt.figure(figsize=(10, 5))

#wykres efektywnosci kulmulatywnej
plt.plot(classifier_names, cumulative_effectiveness, marker='o', label='Skumulowana skuteczność')

#wykres dla pojedynczej efektywnosci
plt.plot(classifier_names, list(effectiveness.values()), marker='o', linestyle='--', label='Oddzielna skuteczność')

#tytuly i etykiety
plt.title('Porównanie efektywności modelu')
plt.xlabel('Klasyfikatory')
plt.ylabel('Efektywność')

#legenda
plt.legend()

#wynik
plt.show()
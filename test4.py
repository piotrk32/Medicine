import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Załadowanie danych
file_path = 'australian.dat'
data = np.loadtxt(file_path)
X, y = data[:, :-1], data[:, -1]

# Normalizacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Trenowanie RandomForestClassifier do oceny ważności cech
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
importances = forest.feature_importances_

# Sortowanie indeksów cech według ważności
indices = np.argsort(importances)[::-1]

# Trenowanie i testowanie modeli dla poszczególnych cech
individual_accuracies = []
for i in indices:
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train[:, [i]], y_train)
    individual_accuracies.append(knn.score(X_test[:, [i]], y_test))

# Trenowanie i testowanie modeli dla kombinacji atrybutów
cumulative_accuracies = []
for i in range(1, len(indices) + 1):
    selected_features = indices[:i]
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train[:, selected_features], y_train)
    cumulative_accuracies.append(knn.score(X_test[:, selected_features], y_test))

# Rysowanie wykresu
plt.figure(figsize=(14, 7))
plt.plot(individual_accuracies, marker='o', linestyle='-', label='Individual Attributes')
plt.plot(cumulative_accuracies, marker='o', linestyle='-', label='Cumulative Attributes')
plt.title('Attribute Ranking and Classification Performance')
plt.xlabel('Ranked Attribute Index')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(range(X.shape[1]), labels=[f"Atr {i+1}" for i in range(X.shape[1])])  # Dodaj etykiety dla atrybutów
plt.show()

# Funkcja do obliczania czułości i specyficzności
def calculate_sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    return sensitivity, specificity

# Obliczanie i zapisywanie metryk dla każdego atrybutu indywidualnie
individual_sensitivities = []
individual_specificities = []
for i in indices:
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train[:, [i]], y_train)
    y_pred = knn.predict(X_test[:, [i]])
    sens, spec = calculate_sensitivity_specificity(y_test, y_pred)
    individual_sensitivities.append(sens)
    individual_specificities.append(spec)

# Obliczanie i zapisywanie metryk dla kumulatywnych atrybutów
cumulative_sensitivities = []
cumulative_specificities = []
for i in range(1, len(indices) + 1):
    selected_features = indices[:i]
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train[:, selected_features], y_train)
    y_pred = knn.predict(X_test[:, selected_features])
    sens, spec = calculate_sensitivity_specificity(y_test, y_pred)
    cumulative_sensitivities.append(sens)
    cumulative_specificities.append(spec)

# Rysowanie wykresu dla czułości
plt.figure(figsize=(14, 7))
plt.plot(individual_sensitivities, marker='o', linestyle='-', label='Individual Sensitivity')
plt.plot(cumulative_sensitivities, marker='o', linestyle='-', label='Cumulative Sensitivity')
plt.title('Individual and Cumulative Sensitivity')
plt.xlabel('Ranked Attribute Index')
plt.ylabel('Sensitivity')
plt.legend()
plt.grid(True)
plt.xticks(range(X.shape[1]), labels=[f"Atr {i+1}" for i in range(X.shape[1])])
plt.show()

# Rysowanie wykresu dla specyficzności
plt.figure(figsize=(14, 7))
plt.plot(individual_specificities, marker='o', linestyle='-', label='Individual Specificity')
plt.plot(cumulative_specificities, marker='o', linestyle='-', label='Cumulative Specificity')
plt.title('Individual and Cumulative Specificity')
plt.xlabel('Ranked Attribute Index')
plt.ylabel('Specificity')
plt.legend()
plt.grid(True)
plt.xticks(range(X.shape[1]), labels=[f"Atr {i+1}" for i in range(X.shape[1])])
plt.show()

# Funkcja do obliczania efektywności
def calculate_effectiveness(sensitivity, specificity):
    return (sensitivity + specificity) / 2

# Obliczanie efektywności dla każdego atrybutu indywidualnie
individual_effectiveness = [calculate_effectiveness(sens, spec) for sens, spec in zip(individual_sensitivities, individual_specificities)]

# Obliczanie efektywności dla kumulatywnych atrybutów
cumulative_effectiveness = [calculate_effectiveness(sens, spec) for sens, spec in zip(cumulative_sensitivities, cumulative_specificities)]

# Rysowanie wykresu dla efektywności
plt.figure(figsize=(14, 7))
plt.plot(individual_effectiveness, marker='o', linestyle='-', label='Individual Effectiveness')
plt.plot(cumulative_effectiveness, marker='o', linestyle='-', label='Cumulative Effectiveness')
plt.title('Individual and Cumulative Effectiveness')
plt.xlabel('Ranked Attribute Index')
plt.ylabel('Effectiveness')
plt.legend()
plt.grid(True)
plt.xticks(range(X.shape[1]), labels=[f"Atr {i+1}" for i in range(X.shape[1])])
plt.show()







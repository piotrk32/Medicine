import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# Przykładowe dane
file_path = 'australian.dat'
data = np.loadtxt(file_path)

# Rozdzielanie danych na cechy i etykiety
X, y = data[:, :-1], data[:, -1]

# Normalizacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Lista do przechowywania wyników skuteczności
individual_accuracies = []
cumulative_accuracies = []

# Trenowanie i testowanie modelu dla każdego atrybutu indywidualnie
for i in range(X.shape[1]):
    # Trenowanie modelu k-NN tylko z i-tym atrybutem
    knn = KNeighborsClassifier(metric='euclidean')
    knn.fit(X_train[:, i:i+1], y_train)
    y_pred = knn.predict(X_test[:, i:i+1])
    accuracy = accuracy_score(y_test, y_pred)
    individual_accuracies.append(accuracy)

# Trenowanie i testowanie modelu kumulatywnie
for i in range(X.shape[1]):
    # Trenowanie modelu k-NN z atrybutami od 0 do i
    knn = KNeighborsClassifier(metric='euclidean')
    knn.fit(X_train[:, :i+1], y_train)
    y_pred = knn.predict(X_test[:, :i+1])
    accuracy = accuracy_score(y_test, y_pred)
    cumulative_accuracies.append(accuracy)

# Rysowanie wykresu
plt.figure(figsize=(10, 5))
plt.plot(range(1, X.shape[1] + 1), individual_accuracies, marker='o', linestyle='-', label='Individual Accuracy')
plt.plot(range(1, X.shape[1] + 1), cumulative_accuracies, marker='o', linestyle='-', label='Cumulative Accuracy')
plt.title('Individual and Cumulative Attribute Accuracies')
plt.xlabel('Number of Attributes')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


def calculate_sensitivity_specificity(y_true, y_pred):
    # Obliczanie macierzy pomyłek
    conf_matrix = confusion_matrix(y_true, y_pred)
    # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    # Obliczanie czułości i specyficzności
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return sensitivity, specificity


# Przygotowanie list na wyniki
sensitivities = []
specificities = []

# Obliczanie metryk dla modeli
for i in range(X_train.shape[1]):
    knn = KNeighborsClassifier(metric='euclidean')
    knn.fit(X_train[:, :i + 1], y_train)
    y_pred = knn.predict(X_test[:, :i + 1])

    sensitivity, specificity = calculate_sensitivity_specificity(y_test, y_pred)
    sensitivities.append(sensitivity)
    specificities.append(specificity)

# Rysowanie wykresu
plt.figure(figsize=(14, 7))
plt.plot(range(1, X_train.shape[1] + 1), sensitivities, marker='o', linestyle='-', label='Sensitivity')
plt.plot(range(1, X_train.shape[1] + 1), specificities, marker='o', linestyle='-', label='Specificity')
plt.plot(range(1, X_train.shape[1] + 1), cumulative_accuracies, marker='o', linestyle='-', label='Cumulative Accuracy')

plt.title('Performance Metrics by Number of Attributes')
plt.xlabel('Number of Attributes')
plt.ylabel('Metric Value')
plt.xticks(range(1, X_train.shape[1] + 1))  # Ensure all attribute numbers are shown
plt.legend()
plt.show()
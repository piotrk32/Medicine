import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Załadowanie danych
# Zakładam, że dane są w formacie, gdzie ostatnia kolumna to etykieta
file_path = 'australian.dat'
data = np.loadtxt(file_path)
X, y = data[:, :-1], data[:, -1]


# Normalizacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


# Definicja funkcji do obliczania metryk
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    return sensitivity, specificity, accuracy


# Trenowanie modeli i obliczanie metryk dla każdego atrybutu oraz kumulatywnie
individual_sensitivities, cumulative_sensitivities = [], []
individual_specificities, cumulative_specificities = [], []
individual_accuracies, cumulative_accuracies = [], []

for i in range(1, X_train.shape[1] + 1):
    knn = KNeighborsClassifier(metric='euclidean')
    knn.fit(X_train[:, :i], y_train)
    y_pred = knn.predict(X_test[:, :i])

    # Indywidualne metryki
    sensitivity, specificity, accuracy = calculate_metrics(y_test, y_pred)
    individual_sensitivities.append(sensitivity)
    individual_specificities.append(specificity)
    individual_accuracies.append(accuracy)

    # Kumulatywne metryki
    if i == 1:
        cumulative_sensitivities.append(sensitivity)
        cumulative_specificities.append(specificity)
        cumulative_accuracies.append(accuracy)
    else:
        cumulative_sensitivities.append((cumulative_sensitivities[-1] * (i - 1) + sensitivity) / i)
        cumulative_specificities.append((cumulative_specificities[-1] * (i - 1) + specificity) / i)
        cumulative_accuracies.append((cumulative_accuracies[-1] * (i - 1) + accuracy) / i)


# Funkcja do rysowania wykresów
def plot_metric(individual_metric, cumulative_metric, metric_name):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(individual_metric) + 1), individual_metric, marker='o', linestyle='-',
             label=f'Individual {metric_name}')
    plt.plot(range(1, len(cumulative_metric) + 1), cumulative_metric, marker='o', linestyle='-',
             label=f'Cumulative {metric_name}')
    plt.title(f'{metric_name} by Number of Attributes')
    plt.xlabel('Number of Attributes')
    plt.ylabel(metric_name)
    plt.xticks(range(1, X_train.shape[1] + 1))
    plt.legend()
    plt.show()

def calculate_effectiveness(sensitivity, specificity):
    return (sensitivity + specificity) / 2
individual_effectiveness, cumulative_effectiveness = [], []

for i in range(1, X_train.shape[1] + 1):
    knn = KNeighborsClassifier(metric='euclidean')
    knn.fit(X_train[:, :i], y_train)
    y_pred = knn.predict(X_test[:, :i])

    # Indywidualne metryki
    sensitivity, specificity, _ = calculate_metrics(y_test, y_pred)
    effectiveness = calculate_effectiveness(sensitivity, specificity)
    individual_effectiveness.append(effectiveness)

    # Kumulatywne metryki
    if i == 1:
        cumulative_effectiveness.append(effectiveness)
    else:
        cumulative_effectiveness.append((cumulative_effectiveness[-1] * (i - 1) + effectiveness) / i)

# Rysowanie wykresów dla każdej metryki
plot_metric(individual_sensitivities, cumulative_sensitivities, 'Sensitivity')
plot_metric(individual_specificities, cumulative_specificities, 'Specificity')
plot_metric(individual_accuracies, cumulative_accuracies, 'Accuracy')
plot_metric(individual_effectiveness, cumulative_effectiveness, 'Effectiveness')
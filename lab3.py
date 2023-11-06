import re
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC

### Ochylenie standardowe


def find_data_start_and_separator(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if re.match(r'^[0-9]', line):
            data_start_line = i
            break
    else:
        raise ValueError('Nie znaleziono linii z danymi numerycznymi.')

    for sep in ['\t', ';', ',', ' ']:
        if sep in lines[data_start_line]:
            separator = sep
            break
    else:
        raise ValueError('Nie znaleziono popularnego separatora w linii z danymi.')

    return data_start_line, separator


file_path = 'C:\\Users\\local\\Downloads\\australian.tab'
data_start_line, separator = find_data_start_and_separator(file_path)

data = pd.read_csv(file_path, sep=separator, header=None, skiprows=data_start_line)

X = data.iloc[:, :-1]  # cechy
y = data.iloc[:, -1]   # etykiety

# Definicja klasyfikatora SVM
svm_classifier = SVC(kernel='linear')

# Przygotowanie procedury walidacji krzyżowej
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Przeprowadzenie walidacji krzyżowej
scores = cross_val_score(svm_classifier, X, y, cv=cv, n_jobs=-1)

# Obliczenie średniej dokładności i odchylenia standardowego
mean_accuracy = np.mean(scores)
std_accuracy = np.std(scores)

# Wyświetlenie wyników
print(f'Średnia dokładność z walidacji krzyżowej: {mean_accuracy * 100:.2f}%')
print(f'Odchylenie standardowe dokładności: {std_accuracy * 100:.2f}%')
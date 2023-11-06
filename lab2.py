import pandas as pd
import re
import scipy.stats as stats  # Zmieniono 'scipi' na 'scipy' i dodano '.stats'
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

def find_data_start_and_separator(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Znajdź pierwszą linię, która wygląda na dane (zakładając, że dane są numeryczne)
    for i, line in enumerate(lines):
        if re.match(r'^[0-9]', line):
            data_start_line = i
            break
    else:
        raise ValueError('Nie znaleziono linii danych.')

    # Znajdź separator, zakładając, że jest to jeden z popularnych separatorów
    for sep in ['\t', ';', ',', ' ']:
        if sep in lines[data_start_line]:
            separator = sep
            break
    else:
        raise ValueError('Nie znaleziono separatora.')

    return data_start_line, separator

file_path = 'C:\\Users\\local\\Downloads\\australian.tab'
data_start_line, separator = find_data_start_and_separator(file_path)
print(f'Dane zaczynają się w linii {data_start_line} i separator to {repr(separator)}.')

# Teraz możemy wczytać dane z pliku, pomijając linie metadanych
data = pd.read_csv(file_path, sep=separator, header=None, skiprows=data_start_line)
print(data.head())

# Podziel dane na cechy (X) i etykiety (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Podziel dane na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utwórz i wytrenuj klasyfikator SVM
svm_classifier = SVC(kernel='linear')  # SVM/ MK
svm_classifier.fit(X_train, y_train)

# Dokonaj predykcji na zestawie testowym
y_pred = svm_classifier.predict(X_test)


# Oblicz i wydrukuj dokładność
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność: {accuracy * 100:.2f}%')

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion matrix:\n{cm}', sep='\n')

# Obliczanie specyficzności
TN = cm[0, 0]
FP = cm[0, 1]
specificity = TN / (TN + FP)

# Obliczanie średniej z dokładności i specyficzności
mean_of_accuracy_and_specificity = (accuracy + specificity) / 2

print(f'Specyficzność: {specificity * 100:.2f}%')
print(f'Średnia z dokładności i specyficzności: {mean_of_accuracy_and_specificity * 100:.2f}%')

#######################################################################################################
# # Używamy całego zestawu danych do walidacji krzyżowej
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]
#
# # Utworzenie klasyfikatora SVM
# svm_classifier = SVC(kernel='linear')
#
# # Przeprowadzamy walidację krzyżową, np. z 10 podziałami
# scores = cross_val_score(svm_classifier, X, y, cv=10)
#
# # Obliczamy średnią i odchylenie standardowe dokładności
# mean_accuracy = np.mean(scores)
# std_accuracy = np.std(scores)
#
# print(f'Średnia dokładność z walidacji krzyżowej: {mean_accuracy * 100:.2f}%')
# print(f'Odchylenie standardowe dokładności: {std_accuracy * 100:.2f}%')



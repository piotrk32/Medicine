import re
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

### standaryzacja i normalizacja
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


# Ustal, które kolumny są cechami (zakładając, że ostatnia kolumna to etykieta)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def standardize(X_train, X_test):
    mean = X_train.mean()
    std = X_train.std()
    X_train_standardized = (X_train - mean) / std
    X_test_standardized = (X_test - mean) / std
    return X_train_standardized, X_test_standardized

# Funkcja do normalizacji danych
def normalize(X_train, X_test):
    min_val = X_train.min()
    max_val = X_train.max()
    X_train_normalized = (X_train - min_val) / (max_val - min_val)
    X_test_normalized = (X_test - min_val) / (max_val - min_val)
    return X_train_normalized, X_test_normalized


# Standaryzuj dane
X_train_standardized, X_test_standardized = standardize(X_train, X_test)

# Normalizuj dane
X_train_normalized, X_test_normalized = normalize(X_train, X_test)

# Wybierz zestaw danych do trenowania i testowania modelu
# Możesz wybrać standaryzowane lub znormalizowane dane
X_train_selected = X_train_standardized
X_test_selected = X_test_standardized

# Utwórz i wytrenuj klasyfikator SVM na wybranym zestawie danych
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_selected, y_train)

# Dokonaj predykcji na zestawie testowym
y_pred = svm_classifier.predict(X_test_selected)

# Oblicz i wydrukuj dokładność
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność: {accuracy * 100:.2f}%')
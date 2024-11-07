import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report
from skimage import io, filters, transform

# Primul pas il reprezinta incarcarea datelor de care vom avea nevoie pentru a antrena si testa modelul
# Folosim functia imRead pentru a incarca imaginile intr-o variabila, pe care ulterior vom realiza si normalizarea datelor:
# Din imagini de 224 x 224px, le vom da resize pe dimensiunea 125 x 125px
# Vom volosi functia cv2.cvtColor pentru a transforma imaginea in din formatul cu 3 canale (R - G - B ) in formatul cu un singur canal ( grayscale ).
# La final, incarcam arrayul imaginilor de test intr-un np array care va stoca setul de date cu imaginile utilizate pentru antrenare.

image_array = []

for i in range (1,15001):
  image = cv2.imread(f"kaggleData/data/{i:06d}.png")
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  resized_image = transform.resize(image_gray,(125,125))
  image_array.append(resized_image)
training_data = np.array(image_array)
image_array.clear()

# La fel ca pasul anterior, repetam acelasi procedeu, in care vom stoca setul de date cu imaginile utilizate pentru validare.

for i in range (15001,17001):
  image = cv2.imread(f"kaggleData/data/{i:06d}.png")
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  resized_image = transform.resize(image_gray,(125,125))
  image_array.append(resized_image)
validation_data = np.array(image_array)
image_array.clear()

# La fel ca pasul anterior, repetam acelasi procedeu, in care vom stoca setul de date cu imaginile utilizate pentru testare.

for i in range (17001,22150):
  image = cv2.imread(f"kaggleData/data/{i:06d}.png")
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  resized_image = transform.resize(image_gray,(125,125))
  image_array.append(resized_image)
test_data = np.array(image_array)

# print(training_data.shape, validation_data.shape, test_data.shape)
# Verificam ca seturile de date au formatul corect

# La pasul urmator, vom incarca etichetele seturilor de date pentru antrenare si validare

with open('kaggleData/train_labels.txt', 'r') as f:
    lines = f.readlines()[1:]
train_labels = np.zeros(len(lines), dtype=int)
for i, line in enumerate(lines):
    line = line.strip().split(',')
    train_labels[i] = int(line[1])

with open('kaggleData/validation_labels.txt', 'r') as f:
    lines = f.readlines()[1:]
validation_labels = np.zeros(len(lines), dtype=int)
for i, line in enumerate(lines):
    line = line.strip().split(',')
    validation_labels[i] = int(line[1])

# Urmatorul proces il reprezinta undersamplingul, pentru a balansa setul de date
# Vom selecta din etichetele datelor de antrenare toate imaginile cu labelul 1, si doar 2000 dintre cele cu labelul 0

label_0 = np.where(train_labels == 0)[0]
label_1 = np.where(train_labels == 1)[0]
undersampled_label_0 = label_0[:2000]

# Vom reface vectorul de imagini astfel incat el sa contina doar cele 2000 de imagini cu labelul 0, si toate cu labelul 1, apoi si vectorul de labeluri pentru datele de antrenare

undersampled_data = np.concatenate((training_data[undersampled_label_0], training_data[label_1]), axis=0)
undersampled_labels = np.concatenate((train_labels[undersampled_label_0], train_labels[label_1]), axis=0)

# Vom folosi functia reshape pentru a transforma un np array din formatul 4D (nrImagini, lungime, latime, canale ) in formatul 2D ( nrImagini, lungimea x latimea imaginii )

# In noul vector de imagini, fiecare linie va reprezenta o imagine, iar fiecare coloana va stoca valorile pixelilor unei imagini
train_flattened = undersampled_data.reshape((4238, 15625))
validation_flattened = validation_data.reshape((2000,15625))
test_flattened = test_data.reshape((5149,15625))

# Importam modelul si il antrenam pe setul de date de antrenare

rfc = RandomForestClassifier(n_estimators=50,
                              max_depth=5,
                              max_features='sqrt',
                              class_weight='balanced')
rfc.fit(train_flattened, undersampled_labels)

# Utilizam setul de validare pentru a face o predictie si a testa diferite scoruri ale modulului pe datele de validare
# Din libraria sklearn.metrics vom folosi functiile accuracy_score, precision, recall si confusion matrix pentru a putea vedea rezultatele modelului pe setul de validare

valid_predictions = rfc.predict(validation_flattened)

accuracy = accuracy_score(validation_labels, valid_predictions)
precision = precision_score(validation_labels, valid_predictions)
recall = recall_score(validation_labels, valid_predictions)
cm = confusion_matrix(validation_labels, valid_predictions)
f1 = f1_score(validation_labels,valid_predictions)
report = classification_report(validation_labels,valid_predictions)

print("F1 Score:\n",f1)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)

# Vom face un nou apel cu functia predict, de data aceasta pe setul de date pentru testare
test_predictions = rfc.predict(test_flattened)

# Vom scrie vectorul de etichete intr-un fisier de tip csv pentru a putea incarca rezultatul pe kaggle
# Prima data vom scrie un header ce va contine id, class, apoi id-ul imaginii si eticheta prezisa

cnt = 17001
csv_filename = 'newOutput.csv'
with open(csv_filename, 'w') as f:
    f.write('id,class\n')
with open(csv_filename, 'a') as f:
    for element in test_predictions:
      f.write('{},{}\n'.format(str(cnt).zfill(6), element))
      cnt += 1

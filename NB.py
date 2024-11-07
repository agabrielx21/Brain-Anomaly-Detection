import numpy as np
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve, auc
from skimage import io, filters, transform
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# Primul pas il reprezinta incarcarea datelor de care vom avea nevoie pentru a antrena si testa modelul
# Folosim functia imRead pentru a incarca imaginile intr-o variabila, pe care ulterior vom realiza si normalizarea datelor:
# Din imagini de 224 x 224px, le vom da resize pe dimensiunea 100 x 100px
# Vom volosi functia cv2.cvtColor pentru a transforma imaginea in din formatul cu 3 canale (R - G - B ) in formatul cu un singur canal ( grayscale ).
# La final, incarcam arrayul imaginilor de test intr-un np array care va stoca setul de date cu imaginile utilizate pentru antrenare.

image_array = []

for i in range (1,15001):
  image = cv2.imread(f"kaggleData/data/{i:06d}.png")
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  resized_image = transform.resize(image_gray,(100,100))
  image_array.append(resized_image)
training_data = np.array(image_array)
image_array.clear()

# La fel ca pasul anterior, repetam acelasi procedeu, in care vom stoca setul de date cu imaginile utilizate pentru validare.

for i in range (15001,17001):
  image = cv2.imread(f"kaggleData/data/{i:06d}.png")
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  resized_image = transform.resize(image_gray,(100,100))
  image_array.append(resized_image)
validation_data = np.array(image_array)
image_array.clear()

# La fel ca pasul anterior, repetam acelasi procedeu, in care vom stoca setul de date cu imaginile utilizate pentru testare.

for i in range (17001,22150):
  image = cv2.imread(f"kaggleData/data/{i:06d}.png")
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  resized_image = transform.resize(image_gray,(100,100))
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

# Vom folosi functia reshape pentru a transforma un np array din formatul 4D (nrImagini, lungime, latime, canale ) in formatul 2D ( nrImagini, lungimea x latimea imaginii )
# In noul vector de imagini, fiecare linie va reprezenta o imagine, iar fiecare coloana va stoca valorile pixelilor unei imagini

train_flattened = training_data.reshape((15000, 10000))
validation_flattened = validation_data.reshape((2000,10000))
test_flattened = test_data.reshape((5149,10000))

# Importam modelul si il antrenam pe setul de date de antrenare

nb = MultinomialNB(force_alpha=True, alpha = 0.1)
nb.fit(train_flattened,train_labels)

# Utilizam setul de validare pentru a face o predictie si a testa diferite scoruri ale modulului pe datele de validare
# Din libraria sklearn.metrics vom folosi functiile accuracy_score, precision, recall si confusion matrix pentru a putea vedea rezultatele modelului pe setul de validare

prediction_labels = nb.predict(validation_flattened)

cm = confusion_matrix(validation_labels, prediction_labels)
report = classification_report(validation_labels, prediction_labels)
f1 = f1_score(validation_labels,prediction_labels)

print("F1 Score:\n",f1)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# Plotam curba Precision-Recall
precision, recall, thresholds = precision_recall_curve(validation_labels,prediction_labels)
auprc = auc(recall, precision)

plt.plot(recall, precision, label='Precision-Recall curve (AUPRC = {:.2f})'.format(auprc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.show()

test_predictions = nb.predict(test_flattened)
# Facem o noua predictie pe baza setului de date pentru testare

# Vom scrie vectorul de etichete intr-un fisier de tip csv pentru a putea incarca rezultatul pe kaggle
# Prima data vom scrie un header ce va contine id, class, apoi id-ul imaginii si eticheta prezisa

cnt = 17001
csv_filename = 'output.csv'
with open(csv_filename, 'w') as f:
    f.write('id,class\n')
with open(csv_filename, 'a') as f:
    for element in test_predictions:
      f.write('{},{}\n'.format(str(cnt).zfill(6), element))
      cnt += 1
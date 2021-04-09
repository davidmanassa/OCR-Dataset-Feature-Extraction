#  IMPORTS
import random

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import feature_extraction as fe


##### HYPER Parametros
training_size = 0.6 # tamanho do conjunto de treino [0-1]


##### READ FILE
file = open("OCRDataSet\\OCRDataSet.txt", "r")

data = []
for str in file.read().split('\t'):
    try:
        data.append(int(str))
    except:
        print("Impossivel de ler inteiro: " + str.__repr__())


##### DATA SPLIT
# Matrix em que cada linha será uma letra
letters = [data[i : i + (50*50)] for i in range(0, len(data), (50*50))]

# Dividar dados (training_size)
count_t_d = int(20 * training_size)

# Shuffle data
tmp = []
for i in range(0, len(letters), 20):
    lst = letters[i : i + 20]
    random.shuffle(lst)
    tmp.extend(lst)
letters = tmp

# Matriz: cada linha é uma letra (36 classes)
training_data = [letters[i : i + count_t_d] for i in range(0, len(letters), 20)]
test_data = [letters[i : i + (20 - count_t_d)] for i in range(count_t_d, len(letters), 20)]

##### Carateristicas (FEATURES)
# -> numero de pixeis a preto
# -> Gaborr bank 

# AKA features
x_train, y_train = [], [] # x_train: nº de imagens de treino linhas, k (nº de carateristicas) carateristicas colunas
# y_train: labels corretos para cada linha
y_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # AKA y_train

# Obter features e construir os dados de treino
for i in range(36):
    feat, labels = fe.getFeaturesLst(training_data[i], y_labels[i])
    x_train.extend(feat)
    y_train.extend(labels)

##### TEST DATA

x_test, y_test = [], []

for i in range(36):
    feat, labels = fe.getFeaturesLst(test_data[i], y_labels[i])
    x_test.extend(feat)
    y_test.extend(labels)

###### TREINAR E CLASSIFICAR
##### SVM

clf = svm.SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

##### DECISION TREE

clf1 = tree.DecisionTreeClassifier()
clf1.fit(x_train, y_train)

y_pred1 = clf1.predict(x_test)

##### KNN

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

y_pred2 = neigh.predict(x_test)


##### ESTATISTICA

def printStatistic(name_algorithm, y_pred, y_test, showHeatMap=False):
    cf_matrix = confusion_matrix(y_test, y_pred.tolist())
    print("Accuracy (%s): %f" % (name_algorithm, accuracy_score(np.array(y_test), y_pred)))
    print("F1-score (%s) por classe:" % name_algorithm)
    print(f1_score(y_test, y_pred.tolist(), average=None))
    print("F1-score (%s) (média pesada): %f" % (name_algorithm, f1_score(y_test, y_pred.tolist(), average='weighted')))
    print(" ")
    if (showHeatMap):
        sns.heatmap(cf_matrix, annot=True)
        plt.show()

printStatistic("SVM", y_pred, y_test, True)
printStatistic("Decision Tree", y_pred1, y_test, True)
printStatistic("KNN", y_pred2, y_test, True)

##### VER FILTROS/LETRAS

# Dividir lista da letra pelas linhas (50x50)
letter_number = 20 * 18 + 10
letter = [letters[letter_number][i : i + 50] for i in range(0, len(letters[letter_number]), 50)]

# imgplot = plt.imshow(letter, cmap='gray')
# plt.show()

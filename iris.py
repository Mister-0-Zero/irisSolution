import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
import math

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)

def irisSolutionInternalTools():
    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    # cm = confusion_matrix(y_test, y_pred)
    #
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    # disp.plot(cmap='Blues')
    #
    # plt.show()

def mySolution():
    class MyKNeighborsClassifier:
        def __init__(self, kol_N):
            self.kol_n = kol_N

        def fit(self, sings, target):
            self.sings = sings
            self.target = target

        def calculationDistance(self, sign, chekingObject):
            return np.sqrt(np.sum((np.array(sign) - np.array(chekingObject)) ** 2))

        def findNeighbors(self, chekingObject):
            mas = []
            for sign, target in zip(self.sings, self.target):
                distance = self.calculationDistance(sign, chekingObject)
                mas.append([distance, target])

            return mas

        def identification_of_results(self, chekingObject):
            mas = self.findNeighbors(chekingObject)
            mas_res = sorted(mas, key=lambda x: x[0])
            mas_Neig = mas_res[:self.kol_n]
            mas_classes = [note[1] for note in mas_Neig]

            return mas_classes

        def predict(self, signs_test):
            predict_mas = []
            for chekingObject in signs_test:
                mas_classes = self.identification_of_results(chekingObject)
                flag = dict()
                for i in mas_classes:
                    if i not in flag:
                        flag[i] = 1
                    else:
                        flag[i] += 1
                predict = -1
                maxValue = 0
                for key, value in flag.items():
                    if maxValue < value:
                        predict = key
                        maxValue = value
                predict_mas.append(predict)

            return np.array(predict_mas)

    model = MyKNeighborsClassifier(kol_N=4)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    def findMistake(X, Y):
        try:
            mistake = dict()
            for i in set(Y):
                mistake[int(i)] = 0
            for x, y in zip(X, Y):
                if x != y:
                    mistake[int(x)] += 1
            print(mistake)
        except Exception as e:
            print(f"Ошибка в findMistake: {e}")

    mistake = findMistake(y_test, y_pred)

    print("Accuracy:", accuracy_score(y_test, y_pred))

irisSolutionInternalTools()
mySolution()


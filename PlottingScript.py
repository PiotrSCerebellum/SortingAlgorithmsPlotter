# Libraries for plotting
import math
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import numpy as np
import timeit
import scipy


def mergeSort(array):
    if len(array) > 1:
        #  r to wskaźnik w którym dzielimy tablice
        r = len(array) // 2
        L = array[:r]
        M = array[r:]
        # Rekursja na podzielonych tablicach
        mergeSort(L)
        mergeSort(M)
        i = j = k = 0  # Indeksy L,M,A
        # Dopóki nie dojdziemy do końca L lub M, wybierz większy
        # element z L i M i przestaw go do odpowiedniego miejsca w A[p..r]
        while i < len(L) and j < len(M):
            if L[i] < M[j]:
                array[k] = L[i]
                i += 1
            else:
                array[k] = M[j]
                j += 1
            k += 1

        # Po przejściu do końca tablicy L lub M
        # wstaw pozostałe elementy do A[p..r]
        while i < len(L):
            array[k] = L[i]
            i += 1
            k += 1

        while j < len(M):
            array[k] = M[j]
            j += 1
            k += 1


def bubbleSort(array):
    # pętla przechodząca po wszystkich elementach
    for i in range(len(array)):

        # pętla porównująca aktualny element z pozostałymi
        for j in range(0, len(array) - i - 1):

            # porównanie elementów
            if array[j] > array[j + 1]:
                # zmiana elementów
                temp = array[j]
                array[j] = array[j + 1]
                array[j + 1] = temp


def countingSort(array):
    size = len(array)
    output = [0] * size
    max = 100# ponieważ takie są losowane
    # Stwórz tabelę ilości elementów
    count = [0] * max

    # Policz liczbę danych elementów i zamieść je w tabeli
    for i in range(0, size):
        count[array[i]] += 1

    # Przekalkuluj tabelę ilości elementów, aby uzyskać kumulatywną tabelę
    for i in range(1, max):
        count[i] += count[i - 1]

    # Odszukaj index każdego elementu w orginalnej tabeli
    # i przenieś je do tabeli posortowanej
    i = size - 1
    while i >= 0:
        output[count[array[i]] - 1] = array[i]
        count[array[i]] -= 1
        i -= 1

    # przekopiuj tabele posortowaną do początkowej
    for i in range(0, size):
        array[i] = output[i]

# Funkcje
def funcLinLog(x, a, b):
    if b > 0:
        return a * x * np.log2(x) + b
    else:
        return -1000


def funcExp(x, a, b,c):
    return a * np.power(x,b)+c


def funcLin(x, a, b):
    return a * x + b

def FuncTester(func,results):
    y_pred=[]
    param, param_cov = scipy.optimize.curve_fit(func, xdata=[*probings], ydata=results, maxfev=100000)
    for x in probings:
        y_pred.append(func(x, *param))
    r2 = r2_score(results, y_pred)
    return y_pred,r2,param

results = []
probings = range(1, 1500000, 1000)
#Możliwe mergeSort(array),bubbleSort(array),countingSort(array)
testedAlgorithm='countingSort(array)'
for x in probings:
    repeats = 1
    array = np.random.randint(100, size=x)
    result = timeit.timeit(testedAlgorithm, globals=globals(), number=repeats)
    print(f"n={x}", result / repeats, "sec")
    results.append(result)
try:
    y_pred_lin, r2_lin, param_lin = FuncTester(funcLin, results)
    print(f'Linear a={param_lin[0]}, b={param_lin[1]}, r2={r2_lin}')
    if r2_lin > 0 and r2_lin < 1:
        plt.plot(probings, y_pred_lin, label=f'linear r2={r2_lin:.4f}')
except:
    pass
try:
    y_pred_linlog, r2_linlog, param_linlog = FuncTester(funcLinLog, results)
    print(f'LinLog a={param_linlog[0]}, b={param_linlog[1]}, r2={r2_linlog}')
    if r2_linlog>0 and r2_linlog<1:
        plt.plot(probings, y_pred_linlog, label=f'linlog r2={r2_linlog:.4f}')
except:
    pass
try:
    y_pred_exp, r2_exp, param_exp = FuncTester(funcExp, results)
    print(f'Exp a={param_exp[0]}, b={param_exp[1]}, c={param_exp[1]}, r2={r2_exp}')
    if r2_exp>0 and r2_exp<1:
        plt.plot(probings, y_pred_exp, label=f'exp r2={r2_exp:.4f}')
except:
    pass


# Let's have a look of the data
plt.plot(probings, results, 'b.')
plt.xlabel('n')
plt.ylabel('time')
plt.legend()
plt.show()


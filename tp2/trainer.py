import numpy as np
import csv
from joblib import dump, load
from sklearn import tree
from hu_generator import generate_hu_moments_file

generate_hu_moments_file()

def label_to_int(string_label):
    if string_label == 'square': return 1
    if string_label == 'triangle': return 2
    if string_label == 'star': return 3

    else:
        raise Exception('unkown class_label')

# dataset
X = []

# etiquetas, correspondientes a las muestras
Y = []

# Agarro las cosas en los archivos las guardo en variables y las mando a train data y labels
def load_training_set():
    global X
    global Y
    with open('generated-files/shapes-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop() # saca el ultimo elemento de la lista
            floats = []
            for n in row:
                floats.append(float(n)) # tiene los momentos de Hu transformados a float.
            X.append(np.array(floats, dtype=np.float32)) # momentos de Hu
            Y.append(np.array([label_to_int(class_label)], dtype=np.int32)) # Resultados
            #Valores y resultados se necesitan por separados
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)

load_training_set()

# entrenamiento
sorter = tree.DecisionTreeClassifier().fit(X, Y)

# visualización del árbol de decisión resultante
tree.plot_tree(sorter)

# guarda el modelo en un archivo
dump(sorter, 'filename.joblib')

# en otro programa, se puede cargar el modelo guardado
loadedSorter = load('filename.joblib')

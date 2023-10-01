from joblib import dump, load
from sklearn import tree

# dataset
X = [
 [6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37],
 [6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37, 6.53608067e-04],
 [6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37],
 [1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37, 6.53608067e-04, 6.07480284e-16, 9.67218398e-18],
 [8.60883492e-28, -1.12639633e-37, 6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37],
 [6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37],
 [9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37, 6.53608067e-04, 6.07480284e-16],
]

# etiquetas, correspondientes a las muestras
Y = [1, 1, 1, 2, 2, 3, 3]

# entrenamiento
sorter = tree.DecisionTreeClassifier().fit(X, Y)

# visualización del árbol de decisión resultante
tree.plot_tree(sorter)

# guarda el modelo en un archivo
dump(sorter, 'filename.joblib')

# en otro programa, se puede cargar el modelo guardado
loadedSorter = load('filename.joblib')
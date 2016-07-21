from sklearn.neighbors import KNeighborsClassifier

# caracter
# tempo de precionamento

#trainingGroup = [[1, 0], [1, 1], [1, 2], [1, 3]];
#groupClassification = ["F", "F", "T", "T"];

#neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(trainingGroup, groupClassification) 

#print(neigh.predict([[1, 0.9], [1, 1.1], [1, 1.9]]))

#print(neigh.predict_proba([[1, 0.9], [1, 1.1], [1, 1.9]]))

# caracter e tempo
trainingGroup = [[1, 0.5], [2, 1], [3, 2], [4, 3]];
groupClassification = ["F", "F", "T", "T"];

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(trainingGroup, groupClassification) 
print(neigh.predict([[8, 1], [9, 0.5], [1, 3], [2, 2.2]]))

# caracter1 e 2 e latência
trainingGroup2 = [[1, 2, 1], [2, 1, 1.2], [3, 4, 2.5], [4, 4, 3]];
groupClassification2 = ["F", "F", "T", "T"];

neigh2 = KNeighborsClassifier(n_neighbors=3)
neigh2.fit(trainingGroup2, groupClassification2) 
print(neigh2.predict([[1, 2, 4], [2, 1, 3], [1, 2, 0.5], [4, 3, 1]]))

# tempo pre. e latência
trainingGroup3 = [[0.5, 1], [1, 1.2], [2, 2.5], [3, 3]];
groupClassification3 = ["F", "F", "T", "T"];

neigh3 = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
neigh3.fit(trainingGroup3, groupClassification3) 

print(neigh3.predict([[5, 3], [4, 4], [1, 0.5], [2, 1], [3, 3], [5,1]]))

# tempo e latência em vetores separados
trainingGroup4 = [[0.5, 1], [1, 1.2], [2, 2.5], [3, 3]];
groupClassification4 = ["F", "F", "T", "T"];

neigh4 = KNeighborsClassifier(n_neighbors=3)
neigh4.fit(trainingGroup4, groupClassification4) 

print(neigh4.predict([[5, 3], [4, 4], [1, 0.5], [2, 1]]))
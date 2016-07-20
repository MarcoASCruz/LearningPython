from sklearn.neighbors import KNeighborsClassifier

# caracter
# tempo de precionamento

trainingGroup = [[1, 0], [1, 1], [1, 2], [1, 3]];
groupClassification = ["F", "F", "T", "T"];

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(trainingGroup, groupClassification) 

print(neigh.predict([[1, 0.9], [1, 1.1], [1, 1.9]]))

print(neigh.predict_proba([[1, 0.9], [1, 1.1], [1, 1.9]]))

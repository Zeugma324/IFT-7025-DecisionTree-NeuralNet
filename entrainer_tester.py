import numpy as np
import sys
import load_datasets
#import DecisionTree # importer la classe de l'arbre de décision
import Neural_Net as Neural_Net# importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester
"""
np.random.seed(42)
train_ratio = 0.7
train, train_labels, test, test_labels = load_datasets.load_wine_dataset(train_ratio)
nn = Neural_Net.NeuralNet(input_size=11, output_size=3, hidden_layer_size=20, learning_rate=0.1, epochs=50000)
nn.train(train, train_labels)
predictions = nn.predict(test)
accuracy = nn.evaluate(test, test_labels)


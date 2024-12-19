"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np


# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class NeuralNet: #nom de la class à changer

	def __init__(self, input_size, output_size, hidden_layer_size, learning_rate=0.01, epochs=100):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_layer_size = hidden_layer_size
		self.learning_rate = learning_rate
		self.epochs = epochs

		self.w1 = self.he_et_al_init(input_size, hidden_layer_size)
		self.b1 = np.zeros((1, hidden_layer_size))
		self.w2 = self.he_et_al_init(hidden_layer_size, output_size)
		self.b2 = np.zeros((1, output_size))

		

	def he_et_al_init(self, n_in, n_out):
		return np.random.randn(n_in, n_out) * np.sqrt(2/n_in)

	def sigmoide(self, z):
		z = np.clip(z, -500, 500)
		return 1/(1+np.exp(-z))
	
	def sigmoide_derivate(self, z):
		return self.sigmoide(z)*(1-self.sigmoide(z))
	
	def RElu(self, z):
		return np.maximum(0,z)

	def RElu_derivate(self, z):
		return (z>0).astype(float)

	def tanh(self, z):
		return np.tanh(z)
	
	def tanh_dericate(self, z):
		return 1-np.tanh(z)**2
	
	def MSE(self, reel, predite):
		return np.mean((reel - predite)**2)
	
	def MSE_derivate(self, reel, predite):
		return 2*(predite - reel)

	def softmax(self, v):
		stable = v - np.max(v, axis=1, keepdims=True)
		exps = np.exp(stable)
		return exps / np.sum(exps, axis=1, keepdims=True)


	def one_hot_encode(self, labels, classes):
		one_hot = np.zeros((labels.size, classes))
		one_hot[np.arange(labels.size), labels] = 1
		return one_hot
		
	def softmax_derivate(self, z):
		return z*(1-z)
	
	def cross_entropy(self, z, y):
		loss = -np.sum(y*np.log(np.clip(z, 1e-8, 1)), axis=1)
		return np.mean(loss)
	
	def cross_entropy_derivate(self, z, y):
		return (z-y)/z.shape[0]

	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
		train_labels = train_labels.astype(int)
		train_labels = self.one_hot_encode(train_labels, self.output_size)

		for i in range(self.epochs):

			N = train.shape[0]
			z1 = np.dot(train, self.w1) + self.b1
			a1 = self.RElu(z1)
			z2 = np.dot(a1, self.w2) + self.b2
			a2 = self.softmax(z2)
			
			loss = self.cross_entropy(a2, train_labels)

			dz2 = self.cross_entropy_derivate(a2, train_labels)

			dw2 = np.dot(a1.T, dz2) / N
			db2 = np.sum(dz2) / N

			da1 = np.dot(dz2, self.w2.T)
			dz1 = da1*self.RElu_derivate(z1)
			dw1 = np.dot(train.T, dz1) / N
			db1 = np.sum(dz1) / N

			self.w2 -= self.learning_rate*dw2
			self.b2 -= self.learning_rate*db2
			self.w1 -= self.learning_rate*dw1
			self.b1 -= self.learning_rate*db1

			#print(f"dw1 mean: {np.mean(dw1)}, db1 mean: {np.mean(db1)}")
			#print(f"dw2 mean: {np.mean(dw2)}, db2 mean: {np.mean(db2)}")

			if (i+1)%1000==0:
				print(f"""epoch : {i+1}/{self.epochs}, loss : {loss:.4f}""")



        
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		z1 = np.dot(x, self.w1) + self.b1
		a1 = self.RElu(z1)
		z2 = np.dot(a1, self.w2) + self.b2
		a2 = self.softmax(z2)
		return np.argmax(a2)


	def evaluate(self, X, y):
		"""
		c'est la méthode qui va évaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		preds = [self.predict(X[i:i+1]) for i in range(len(X))]
		acc = np.mean(np.array(preds) == y)

		print(f"acc : {acc:.2f}")
		return acc

	
	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.
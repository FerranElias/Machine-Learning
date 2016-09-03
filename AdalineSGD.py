###Stochastic gradient descent.Instead of updating the weights based on the sum of the accumulated errors over all observations, we update the weights for each training sample.
###Although SGD can be considered as an approximation of gradient descent, it typically reaches convergence much faster because of the more frequent weight updates. Since each gradient
###is calculated based on a single training example, the error surface is noisier than in gradient descent, which can also have the advantage that SGD can escape shallow local 
###minima more readily. To obtain accurate results via SGD, it is important to present it with data in a random order, which is why we want to shuffle the training set for every
###epoch to prevent cycles.



from numpy.random import seed
import numpy as np

class AdalineSGD(object):
	"""ADAptive LInear NEuron classifier.
	
	Parameters
	----------
	eta: float
		learning rate (between 0 and 1)
	n_iter: int
		passes over the training dataset
		
	Attributes
	----------
	w_: 1d-array
		Weights after fitting.
	errors_: list
		Number of misclassifications in every epoch
	shuffle: bool (default: True)
		Shuffles training data every epoch if True to prevent cycles
	random_state: int (default: None)
		Set random state for shuffling and initializing the weights.
	"""
	def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
		self.eta=eta
		self.n_iter=n_iter
		self.w_initialized=False
		self.shuffle=shuffle
		if random_state:
			seed(random_state)
	def fit(self, X, y):
		"""Fit training data.
		
		Parameters
		----------
		X: {array-like}, shape=[n_samples, n_features]
		Training vectors, where n_samples is the number of samples and n_features is the number of features.
		
		y: array-like, shape=[n_samples]
			Target values.
		
		Returns
		-------
		self: object
		"""
		
		self.w_=np.zeros(1+X.shape[1])
		self.cost_=[]
		
		for _ in range(self.n_iter):
			if self.shuffle:
				X,y=self._shuffle(X,y)
			cost=[]
			for xi, target in zip(X,y):
				cost.append(self._update_weights(xi,target))
			avg_cost=sum(cost)/len(y)
			self.cost_.append(avg_cost)
		return self

	def partial_fit(self,X,y):
		"""Fit training data without reinitializing the weights"""
		if not self.w_initialized:
			self._initialize_weights(X.shape[1])
		if y.ravel().shape[0]>1:
			for xi, target in zip(X,y):
				self._update_weights(X,y)
		else:
			self._update_weights(X,y)
		return self
		
	def _shuffle(self,X,y):
		"""Shuffle training data"""
		r=np.random.permutation(len(y))
		return X[r],y[r]
	
	def _initialize_weights(self,m):
		"""Initialize weights to zeros"""
		self.w_=np.zeros(1+m)
		self.w_initialized=True
		
	def _update_weights(self,xi,target):
		"""Apply Adaline learning rule to update the weights"""
		output=self.net_input(xi)
		error=(target-output)
		self.w_[1:]+=self.eta*xi.dot(error)
		self.w_[0]+=self.eta*error
		cost=0.5*error**2
		return cost
		
	def net_input(self,X):
		"""calculate net input"""
		return np.dot(X,self.w_[1:])+self.w_[0]

	def activation(self,X):
		"""Compute linear activation"""
		return self.net_input(X)
		
	def predict(self,X):
		"""Return class label after unit step"""
		return np.where(self.activation(X)>=0.0,1,-1)

import pandas as pd

df=pd.read_csv('E:\Dropbox\Copenhagen\Python\datasets\irisdata.txt', header=None)
#print(df.tail())

import matplotlib.pyplot as plt
import numpy as np

y=df.iloc[0:100,4].values
#print(y)

y=np.where(y=='Iris-setosa',-1,1)
#print(y)

X=df.iloc[0:100, [0,2]].values
#print(X)

plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()




##########next step: standardization
X_std=np.copy(X)
X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()

from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier, resolution=0.02):
	#setup marker generator and color map
	markers=('s','x','o','^','v')
	colors=('red','blue','lightgreen','gray','cyan')
	cmap=ListedColormap(colors[:len(np.unique(y))])
	
	#plot the decision surface
	x1_min, x1_max=X[:,0].min()-1, X[:,0].max()+1
	x2_min, x2_max=X[:,1].min()-1, X[:,1].max()+10
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z=Z.reshape(xx1.shape)
	plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())
	
	#plot class samples
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8, s=cmap(idx), marker=markers[idx], label=cl)
		
ada=AdalineSGD(n_iter=15,eta=0.01, random_state=1)
ada.fit(X_std,y)
plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()



























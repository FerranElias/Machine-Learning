import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wine=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier, test_idx=None, resolution=0.02):
	#setup marker generator and color map
	markers=('s','x','o','^','v')
	colors=('red','blue','lightgreen','gray','cyan')
	cmap=ListedColormap(colors[:len(np.unique(y))])
	
	#plot the decision surface
	x1_min, x1_max=X[:,0].min()-1, X[:,0].max()+1
	x2_min, x2_max=X[:,1].min()-1, X[:,1].max()+1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z=Z.reshape(xx1.shape)
	plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())
	
	#plot all samples
	X_test, y_test=X[test_idx,:],y[test_idx]
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8, s=cmap(idx), marker=markers[idx], label=cl)

from sklearn.cross_validation import train_test_split
X,y=df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)		

#############standardization
from sklearn.preprocessing import StandardScaler
stdsc=StandardScaler()
X_train_std=stdsc.fit_transform(X_train)
X_test_std=stdsc.transform(X_test)	
		
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
lr=LogisticRegression()
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
lr.fit(X_train_pca,y_train)
plot_decision_regions(X_train_pca,y_train,classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

plot_decision_regions(X_test_pca,y_test,classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

#if we are interested in the explained variance ratios of the different principal components, we can simply initialize the PCA class with the n_components parameter set to None,
#so all principal components are kept and the explained variance ratio can then be accessed via the explained_variance_ratio_ attribute
pca=PCA(n_components=None)
X_train_pca=pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)
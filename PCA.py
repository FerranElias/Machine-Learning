import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wine=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
X,y=df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.fit_transform(X_test)

##the eigenvectors of the covariance matrix represent the principal components (the directions of maximum variance), whereas the corresponding eigenvalues will define their magnitude
#now, we obtain the eigenpairs of the covariance matrix
cov_mat=np.cov(X_train_std.T)
eigen_vals, eigen_vecs=np.linalg.eig(cov_mat)
print('\n Eigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

##after we have successfully decomposed the covariance matrix into eigenpairs, let's now proceed with the last three steps to transform the wine dataset onto the new principal
#component axes. now, we will sort the eigenpairs by descending order of the eigenvalues, construct a projection matrix from the selected eigenvectors, and use the projection matrix
#to transform the data onto the lower-dimensional subspace.

eigen_pairs=[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

##next, we collect the two eigenvectors that correspond to the two largest values to capture about 60% of the variance in this dataset. note that we only chose two eigenvectors
#for the purpose of illustration, since we are going to plot the data via a two-dimensional scatter plot later in this subsection. in practice, the number of principal components
#has to be determined from a trade-off between computational efficiency and the performance of the classifier.

w=np.hstack((eigen_pairs[0][1][:,np.newaxis],
			eigen_pairs[1][1][:,np.newaxis]))
print('Matrix W:\n', w)

##by executing the preceding code, we have created a 13x2 dimensional projection matrix W from the top two eigenvectors. using the projection matrix, we can now transform a sample
# x (represented as 1x13-dimensional row vector) onto the PCA subspace obtaining x', a now two-dimensional sample vector consisting of two new features:
X_train_std[0].dot(w)

#similarly, we can transform the entire 124x13-dimensional training dataset onto the two principal components by calculating the matrix dot product.
X_train_pca=X_train_std.dot(w)

#let's visualize the transformed wine dataset, now stored as a 124x2 dimensional matrix, in a two-dimensional scatterplot
colors=['r','b','g']
markers=['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
	plt.scatter(X_train_pca[y_train==l,0],
				X_train_pca[y_train=l,1],
				c=c,label=l,marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
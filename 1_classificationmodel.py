from matplotlib import pyplot as plt
import numpy as np

#we load the data with load_iris from sklearn
from sklearn.datasets import load_iris
data=load_iris()

#load_iris returns an object with several fields
features=data.data
feature_names=data.feature_names
target=data.target
target_names=data.target_names

'''
for t in range(3):
	if t==0:
		c='r'
		marker='>'
	elif t==1:
		c='g'
		marker='o'
	elif t==2:
		c='b'
		marker='x'
	plt.scatter(features[target==t,0],
				features[target==t,1],
				marker=marker,
				c=c)
	plt.show()
				
'''

##classification model

#we use numpy fancy indexing to get an array of strings
labels=target_names[target]

#the petal length is the feature at position 2
plength=features[:,2]

#build an array of booleans:
is_setosa=(labels=='setosa')

max_setosa=plength[is_setosa].max()
min_non_setosa=plength[~is_setosa].min()
print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))

#therefore, we can build a simple model: if the petal length is smaller than 2, then this is an Iris Setosa flower; 
#otherwise it is either Iris Virginica or Iris Versicolor.

#now we try to separate Iris Virginica from Iris Versicolor.
features=features[~is_setosa]
labels=labels[~is_setosa]
#build a new target variable, is_virginica
is_virginica=(labels=='virginica')

#initialize best_acc to impossibly low value
best_acc=-1.0
for fi in range(features.shape[1]):
	#we are going to test all possible thresholds
	thresh=features[:,fi]
	for t in thresh:
		#get the vector for feature 'fi'
		feature_i=features[:,fi]
		#apply threshold 't'
		pred=(feature_i>t)
		acc=(pred==is_virginica).mean()
		rev_acc=(pred==~is_virginica).mean()
		if rev_acc>acc:
			reverse=True
			acc=rev_acc
		else:
			reverse=False
		if acc>best_acc:
			best_acc=acc
			best_fi=fi
			best_t=t
			best_reverse=reverse









import pandas as pd
import numpy as np

df_wine=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns=['Class label', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
				'Color Intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
				
print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())

from sklearn.cross_validation import train_test_split
X,y=df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#############standardization
from sklearn.preprocessing import StandardScaler
stdsc=StandardScaler()
X_train_std=stdsc.fit_transform(X_train)
X_test_std=stdsc.transform(X_test)

##L1 regularization
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1')

lr= LogisticRegression(penalty='l1',C=0.1)
lr.fit(X_train_std,y_train)
print('Training accuracy:', lr.score(X_train_std,y_train))
print('Test accuracy:', lr.score(X_test_std,y_test))
print(lr.intercept_)
print(lr.coef_)

#we plot the regularization path, which is the weight coefficients of the different features features for different regularization strengths
import matplotlib.pyplot as plt
fig=plt.figure()
ax=plt.subplot(111)
colors=['blue','green','red','cyan','magenta','yellow','black','pink','lightgreen','lightblue','gray','indigo','orange']
weights, params=[],[]
for c in np.arange(-4,6):
	lr=LogisticRegression(penalty='l1',C=10**c,random_state=0)
	lr.fit(X_train_std,y_train)
	weights.append(lr.coef_[1])
	params.append(10**c)
weights=np.array(weights)
for column, color in zip(range(weights.shape[1]),colors):
	plt.plot(params,weights[:,column],label=df_wine.columns[column+1],color=color)
plt.axhline(0,color='black',linestyle='--',linewidth=3)
plt.xlim([10**(-5),10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=True)
plt.show()
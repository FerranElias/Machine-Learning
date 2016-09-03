from sklearn.datasets import load_boston
boston=load_boston()

#boston.data contains the input data and boston.target contains the price of houses
#we will start with a simple one-dimensional regression, trying to regress the price on a single attribute, the average number of rooms
#per dwelling in the neighborhood, which is stored at position 5

from matplotlib import pyplot as plt
plt.scatter(boston.data[:,5], boston.target, color='r')
plt.show()

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

x=boston.data[:,5]
y=boston.target
import numpy as np
x=np.transpose(np.atleast_2d(x))	#converts x from a one-dimensional to a two-dimensional array. this conversion is necessary 
#as the fit method expects a two-dimensional array as its first argument.
lr.fit(x,y)
y_predicted=lr.predict(x)

#we measure how good of a fit this is quantitatively
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y, lr.predict(x))
print("Mean squared error (of training data): {:.3}".format(mse))

#this number can sometimes be hard to interpret, and it's better to take the square root, to obtain the root mean square error (RMSE)
rmse=np.sqrt(mse)
print("RMSE (of training data): {:.3}".format(rmse))

#the rmse is still hard to interpret. we can use the coefficient of determination or r_squared
from sklearn.metrics import r2_score
r2=r2_score(y,lr.predict(x))
print("R2 (on training data): {:.2}".format(r2))

##############multidimensional regression
x= boston.data
y=boston.target
lr.fit(x,y)
mse=mean_squared_error(y, lr.predict(x))
print("Mean squared error (of training data): {:.3}".format(mse))
rmse=np.sqrt(mse)
print("RMSE (of training data): {:.3}".format(rmse))
r2=r2_score(y,lr.predict(x))
print("R2 (on training data): {:.2}".format(r2))

#we can plot the prediction versus the actual value
p=lr.predict(x)
plt.scatter(p,y)
plt.xlabel('Predicted price')
plt.ylabel('Actual price')
plt.plot([y.min(), y.max()],[[y.min()], [y.max()]])
plt.show()

####cross-validation for regression
from sklearn.cross_validation import KFold
kf=KFold(len(x), n_folds=5)
p=np.zeros_like(y)
for train, test in kf:
	lr.fit(x[train], y[train])
	p[test]=lr.predict(x[test])
rmse_cv=np.sqrt(mean_squared_error(p,y))
print('RMSE on 5-fold CV: {:.2}'.format(rmse_cv))

###############PENALIZED OR REGULARIZED REGRESSION
"""
L1 penalty (also known as Lasso): we add the absolute value of the betas to the cost function

L2 penalty (also known as Ridge): we add the squared value of the betas to the cost function

Elastic Nets: when we use boths

Both the lasso and the ridge result in smaller coefficients than unpenalized regression (smaller in absolute value, ignoring the sign)
However, the lasso has the additional property that it results in many coefficients being set to exactly zero! This means that the final
model does not even use some of its input features, the model is sparse.
"""
from sklearn.linear_model import ElasticNet, Lasso
en=ElasticNet(alpha=0.5)
x= boston.data
y=boston.target
en.fit(x,y)
mse=mean_squared_error(y, en.predict(x))
print("Mean squared error (of training data): {:.3}".format(mse))
rmse=np.sqrt(mse)
print("RMSE (of training data): {:.3}".format(rmse))
r2=r2_score(y,en.predict(x))
print("R2 (on training data): {:.2}".format(r2))
kf=KFold(len(x), n_folds=5)
p=np.zeros_like(y)
for train, test in kf:
	en.fit(x[train], y[train])
	p[test]=en.predict(x[test])
rmse_cv=np.sqrt(mean_squared_error(p,y))
print('RMSE on 5-fold CV: {:.2}'.format(rmse_cv))

#visualizing the lasso path
las=Lasso(normalize=1)
alphas=np.logspace(-5,2,1000)
alphas, coefs, _=las.path(x,y,alphas=alphas)	#for each value in alphas, the path method on the lasso object returns the coefficients
												#that solve the lasso problem with that parameter value
fix, ax=plt.subplots()
ax.plot(alphas,coefs.T)
ax.set_xscale('log')
ax.set_xlim(alphas.max(), alphas.min())
plt.show()

#################P-GREATER-THAN-N SCENARIOS
from sklearn.datasets import load_svmlight_file
data, target =load_svmlight_file('E2006.train')

#we can start by looking at some attributes of the target
print('Min target value: {}'.format(target.min()))
print('Max target value: {}'.format(target.max()))
print('Mean target value: {}'.format(target.mean()))
print('Std. Dev. target value: {}'.format(target.std()))

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(data,target)
pred=lr.predict(data)
rmse_train=np.sqrt(mean_squared_error(target, pred))
print('RMSE on training: {:.2}'.format(rmse_train))
print('R2 on training: {:.2}'.format(r2_score(target, pred)))
kf=KFold(len(x), n_folds=5)
pred=np.zeros_like(target)
for train, test in kf:
	lr.fit(data[train], target[train])
	pred[test]=lr.predict(data[test])
rmse_cv=np.sqrt(mean_squared_error(pred,target))
print('RMSE on 5-fold CV: {:.2}'.format(rmse_cv))

from sklearn.linear_model import ElasticNet
met=ElasticNet(alpha=0.1)

kf=KFold(len(x), n_folds=5)
pred=np.zeros_like(target)
for train, test in kf:
	met.fit(data[train], target[train])
	pred[test]=met.predict(data[test])
rmse = np.sqrt(mean_squared_error(target, pred))
print('[EN 0.1] RMSE on testing (5 fold): {:.2}'.format(rmse))
r2 = r2_score(target, pred)
print('[EN 0.1] R2 on testing (5 fold): {:.2}'.format(r2))






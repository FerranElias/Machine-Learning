###########setting hyperparameters in a principled way
#before we set the penalty parameter to 0.1. how do we choose a good value? a generic solutions is to use cross-validation
#we pick a set of possible values, and then use cross-validation to choose which one is best
#in order to obtain an estimate of generalization, we have to use two-levels of cross-validation: one level is to estimate the 
#generalization, while the second level is to get good parameters
#fortunately, scikit-learn makes it very easy to do the right thing; it provides classes named LassoCV, RidgeCV, and ElasticNetCV,
#al of which encapsulate an inner cross-validation loop to optimize for the necessary parameter
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.cross_validation import KFold
from sklearn.datasets import load_svmlight_file
data,target = load_svmlight_file('E2006.train')
#met=ElasticNetCV(n_jobs=4)		#make use of four cpus
met=ElasticNetCV()	
kf=KFold(len(target),n_folds=5)
p=np.zeros_like(target)
for train, test in kf:
	met.fit(data[train],target[train])
	p[test]=met.predict(data[test])
r2_cv=r2_score(target,p)
print("R2 ElasticNetCV: {:.2}".format(r2_cv))

#you may have wondered why, if elastic nets have two penalties, we only need to set a single value for alpha
#in fact, the two values are specified by separately specifying alpha and the l1_ratio variable
#alpha_1=rho*alpha and alpha_2=(1-rho)*alpha
#in an intuitive sense, alpha sets the overall amount of regularization while l1_ratio sets the tradeoff between the different 
#types of regularization, L1 and L2.
#we can request that the ElasticNetCV objects tests different values of l1_ratio:
l1_ratio=[.01, .05, .25, .5, .75, .95, .99]]
met=ElasticNetCV(l1_ratio=l1_ratio,n_jobs=-1)	#would use all available cpus
#This set of l1_ratio values is recommended in the documentation. It will test
#models that are almost like Ridge (when l1_ratio is 0.01 or 0.05) as well as models
#that are almost like Lasso (when l1_ratio is 0.95 or 0.99). Thus, we explore a full
#range of different options.

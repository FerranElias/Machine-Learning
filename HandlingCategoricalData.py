import pandas as pd
df = pd.DataFrame([['green','M',10.1,'class1'],
					['red','L',13.5,'class2'],
					['blue','XL',15.3,'class1']])
df.columns=['color','size','price','classlabel']
print(df)

#mapping ordinal features
size_mapping={'XL':3,'L':2,'M':1}
df['size']=df['size'].map(size_mapping)
print(df)

#if we want to transform the integer values back to the original string representation at a later stage, we can simply define a reverse-mapping dictionary
#inv_size_mapping={v:k for k, v in size_mapping.items()} that can be used via the pandas' map method on the transformed feature column similar to the size_mapping dictionary that 
#we used previously

#encoding class labels. remember that class labels are not ordinal, and it doesn't matter which integer number we assign to a particular string label.
import numpy as np
class_mapping={label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)
df['classlabel']=df['classlabel'].map(class_mapping)
print(df)

inv_class_mapping={v:k for k, v in class_mapping.items()}
df['classlabel']=df['classlabel'].map(inv_class_mapping)

#alternatively, there is a convenient label encoder class directly implemented in scikit-learn to achieve the same
from sklearn.preprocessing import LabelEncoder
class_le=LabelEncoder
#y=class_le.fit_transform(df['classlabel'].values)		#the fit_transform method is just a shortcut for calling fit and transform separately.
#print(y)

#we can use the inverse_transform to transform the integer class back into their original string representation
#class_le.inverse_transform(y)

###########performing one-hot encoding on nominal features
#it may appear that we could use the LabelEncoder to transform the nominal color column of our dataset. But if we do that, the learning algorithm would assume that color1 is larger
#than color2 and so on. Although this assumption is incorrect, the algorithm could still produce useful results. However, those results would not be optimal.
#a common workaround for this problem is to use a technique called one-hot encoding. the idea behind this approach is to create a new dummy feature for each unique value in 
#the nominal feature column.

X=df[['color','size','price']].values
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[0])
#print(ohe.fit_transform(X).toarray())

#an even more convenient way to create the dummy features is to use get_dummies method
print(pd.get_dummies(df[['price','color','size']]))
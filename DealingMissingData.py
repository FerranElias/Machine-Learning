
#we enter data
import pandas as pd
from io import StringIO
csv_data='''A,B,C,D
		1.0,2.0,3.0,4.0
		5.0,6.0,,8.0
		0.0,11.0,12.0,'''
df=pd.read_csv(StringIO(csv_data))

print(df.isnull().sum())		#we return the number of missing values per column

#although scikit-learn was developed for working with numpy arrays, it can sometimes be more convenient to preprocess data using pandas' dataframe. 
#we can always access the underlying numpy array of the dataframe via the values attribute before we feed it into a scikit-learn estimator
print(df.values)

#one of the easiest ways to deal with missing data is to simply remove the corresponding features (columns) or samples (rows) from the dataset entirely.
#rows with missing values can be easily dropped via the dropna method
print(df.dropna())		

#similarly, we can drop columns that have at least one NaN in any row by setting the axis argument to 1.
print(df.dropna(axis=1))

#only drop rows where all columns are NaN
print(df.dropna(how='all'))

#drop rows that have not at least 4 non-NaN values
print(df.dropna(thresh=4))

#only drop rows where NaN appear in specific columns (here: 'C')
print(df.dropna(subset=['C']))


##imputing missing values
#in the next lines, we replace each NaN value by the corresponding mean, which is separately calculated for each feature column. If we changed the setting axis=0 to axis=1, we would
#calculate the row means. Other options for the strategy parameter are median or most_frequent, where the latter replaces the missing values by the most frequent values.
from sklearn.preprocessing import Imputer
imr=Imputer(missing_values='NaN',strategy='mean',axis=0)
imr=imr.fit(df)		#the fit method is used to learn the parameters from the training data
imputed_data=imr.transform(df.values)		#the transform method uses those parameters to transform the data
print(imputed_data)





















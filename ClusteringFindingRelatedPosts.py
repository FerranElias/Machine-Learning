from sklearn.feature_extraction.text import CountVectorizer
import scipy as sp

vectorizer=CountVectorizer(min_df=1)		#min_df means that words that occur less than 1 time will be dropped. it could also be a frequency
print(vectorizer)

content=["How to format my hard disk","Hard disk format problems"]
X=vectorizer.fit_transform(content)	#learns a vocabulary dictionary of all tokens in the raw documents and returns a term-dcoument matrix telling you in which document each token occurs
vectorizer.get_feature_names
print(X.toarray().transpose())

#now we start with some posts
#from nltk.corpus import PlaintextCorpusReader
#corpus_root='E:\Dropbox\Copenhagen\Python\BuildingMachineLearningSystems\ClusteringFindingRelatedPosts\posts'
#posts=PlaintextCorpusReader(corpus_root,'.*')

import os
from os import path, listdir
posts=[open(os.path.join('E:\Dropbox\Copenhagen\Python\BuildingMachineLearningSystems\ClusteringFindingRelatedPosts\posts',f)).read() for f in os.listdir('E:\Dropbox\Copenhagen\Python\BuildingMachineLearningSystems\ClusteringFindingRelatedPosts\posts')]

X_train=vectorizer.fit_transform(posts)
num_samples, num_features=X_train.shape
print("#samples: %d, #features: %d" % (num_samples,num_features))
print(vectorizer.get_feature_names())

#now we vectorize our new post
new_post="imaging databases"
new_post_vec=vectorizer.transform([new_post])	#transform documents to document-term matrix. extract token counts out of raw text documents using the vocabulary fitted with fit

print(new_post_vec)		#the count vectors returned are sparse. they will be represented with a coo_matrix ("COOrdinate") that will tell us the coordinate of the words with respect to the vector of words in the training data
print(new_post_vec.toarray())		#via toarray() we can access the full array

#we define a function to calculate the euclidean distance
def dist_raw(v1,v2):
	delta=v1-v2
	return sp.linalg.norm(delta.toarray())		#the function norm() calculates the euclidean norm
	
#we calculate the distance for each post
import sys
best_doc=None
best_dist=sys.maxsize
best_i=None
for i, post in enumerate(range(num_samples)):
	if post==new_post:
		continue
	post_vec=X_train.getrow(i)
	d=dist_raw(post_vec,new_post_vec)
	print("=== Post %i with dist=%.2f: %s"%(i,d,post))
	if d<best_dist:
		best_dist=d
		best_i=i
print("Best post is %i with dist=%.2f"%(best_i,best_dist))

#post 3 and 4 are the same, but post 4 is post 3 duplicated 3 times. it should have the same similarity... we need to normalize
print(X_train.getrow(3).toarray())
print(X_train.getrow(4).toarray())

#####################################we repeat the procedure but with normalized vectors
def dist_norm(v1,v2):
	v1_normalized=v1/sp.linalg.norm(v1.toarray())
	v2_normalized=v2/sp.linalg.norm(v2.toarray())
	delta=v1_normalized-v2_normalized
	return sp.linalg.norm(delta.toarray())
	
#we calculate the distance for each post
import sys
best_doc=None
best_dist=sys.maxsize
best_i=None
for i, post in enumerate(range(num_samples)):
	if post==new_post:
		continue
	post_vec=X_train.getrow(i)
	d=dist_norm(post_vec,new_post_vec)
	print("=== Post %i with dist=%.2f: %s"%(i,d,post))
	if d<best_dist:
		best_dist=d
		best_i=i
print("Best post is %i with dist=%.2f"%(best_i,best_dist))

##########################################now we remove stop words
vectorizer=CountVectorizer(min_df=1, stop_words='english')

X_train=vectorizer.fit_transform(posts)
num_samples, num_features=X_train.shape
print("#samples: %d, #features: %d" % (num_samples,num_features))
print(vectorizer.get_feature_names())

#now we vectorize our new post
new_post="imaging databases"
new_post_vec=vectorizer.transform([new_post])	#transform documents to document-term matrix. extract token counts out of raw text documents using the vocabulary fitted with fit

#we calculate the distance for each post
import sys
best_doc=None
best_dist=sys.maxsize
best_i=None
for i, post in enumerate(range(num_samples)):
	if post==new_post:
		continue
	post_vec=X_train.getrow(i)
	d=dist_norm(post_vec,new_post_vec)
	print("=== Post %i with dist=%.2f: %s"%(i,d,post))
	if d<best_dist:
		best_dist=d
		best_i=i
print("Best post is %i with dist=%.2f"%(best_i,best_dist))

#######################stemming
import nltk.stem
english_stemmer=nltk.stem.SnowballStemmer('english')

#we extend the vectorizer with the nltk stemmer
#what the next lines do: lower casing, extract individual words in tokenization step, stemming the words
#what i understand from the next lines of code is that we include the stemming into the class CountVectorizer, building a sort of superclass
class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer=super(StemmedCountVectorizer,self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
		
vectorizer=StemmedCountVectorizer(min_df=1, stop_words='english')

X_train=vectorizer.fit_transform(posts)
num_samples, num_features=X_train.shape
print("#samples: %d, #features: %d" % (num_samples,num_features))
print(vectorizer.get_feature_names())

#now we vectorize our new post
new_post="imaging databases"
new_post_vec=vectorizer.transform([new_post])	#transform documents to document-term matrix. extract token counts out of raw text documents using the vocabulary fitted with fit

#we calculate the distance for each post
import sys
best_doc=None
best_dist=sys.maxsize
best_i=None
for i, post in enumerate(range(num_samples)):
	if post==new_post:
		continue
	post_vec=X_train.getrow(i)
	d=dist_norm(post_vec,new_post_vec)
	print("=== Post %i with dist=%.2f: %s"%(i,d,post))
	if d<best_dist:
		best_dist=d
		best_i=i
print("Best post is %i with dist=%.2f"%(best_i,best_dist))		

##########counting term frequencies for every post and discounting those that appear in many posts
#we want a high value for a given term in a given value, if that term occurs often in that particular post and very seldom anywhere else
#this is exactly what 'term frequency-inverse document frequency' (TFIDF) does

def tfidf(term,doc,corpus):
	tf=doc.count(term)/len(doc)
	num_docs_with_term=len([d for d in corpus if term in d])
	idf=sp.log(len(corpus)/num_docs_with_term)
	return tf*idf
	
#we can also use the TfidfVectorizer, which is inherited from CountVectorizer
#when we use that, the resulting document vectors will not contain counts any more, but the individual tfidf values per term
from sklearn.feature_extraction.text import TfidfVectorizer
class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer=super(StemmedTfidfVectorizer,self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
vectorizer=StemmedTfidfVectorizer(min_df=1, stop_words='english', decode_error='ignore')
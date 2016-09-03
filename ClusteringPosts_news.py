import nltk.stem
import scipy as sp
english_stemmer=nltk.stem.SnowballStemmer('english')
import sklearn.datasets
all_data=sklearn.datasets.fetch_20newsgroups(subset='all')
print(len(all_data.filenames))
print(all_data.target_names)

#we will limit the data to some groups
groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
train_data=sklearn.datasets.fetch_20newsgroups(subset='train',categories=groups)
test_data=sklearn.datasets.fetch_20newsgroups(subset='test',categories=groups)

from sklearn.feature_extraction.text import TfidfVectorizer
class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer=super(StemmedTfidfVectorizer,self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
vectorizer=StemmedTfidfVectorizer(min_df=1, stop_words='english', decode_error='ignore')

vectorized=vectorizer.fit_transform(train_data.data)
num_samples, num_features=vectorized.shape

num_clusters=50
from sklearn.cluster import KMeans
km=KMeans(n_clusters=num_clusters,init='random',n_init=1,verbose=1,random_state=3)
km.fit(vectorized)

print(km.labels_)
print(km.labels_.shape)
print(km.cluster_centers_)

new_post = \
    """Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now. I tried to format it, but now it doesn't boot any more. Any ideas? Thanks."""

new_post_vec=vectorizer.transform([new_post])
new_post_label=km.predict(new_post_vec)[0]

#we get the indices of the posts of the same cluster as our new posts
similar_indices=(km.labels_==new_post_label).nonzero()[0]		#nonzero() converts a boolean array into a smaller array containing the indices of the true elements

similar=[]
for i in similar_indices:
	dist=sp.linalg.norm((new_post_vec-vectorized[i]).toarray())
	similar.append((dist,train_data.data[i]))
similar=sorted(similar)
print(len(similar))
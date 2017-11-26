import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

vectorizer = TfidfVectorizer(max_features = 4096)
vectors1 = vectorizer.fit_transform(newsgroups_train.data)
vectors2 = vectorizer.fit_transform(newsgroups_test.data)
vectors1 = vectors1.todense()
vectors2 = vectors2.todense()

indices0 = [i for i, x in enumerate(newsgroups_train.target) if x ==0 ]
indices1 = [i for i, x in enumerate(newsgroups_train.target) if x == 1]
indices2 = [i for i, x in enumerate(newsgroups_train.target) if x == 2]
indices3 = [i for i, x in enumerate(newsgroups_train.target) if x == 3]
indices4 = [i for i, x in enumerate(newsgroups_train.target) if x == 4]
indices5 = [i for i, x in enumerate(newsgroups_train.target) if x == 5]
indices6 = [i for i, x in enumerate(newsgroups_train.target) if x == 6]
indices7 = [i for i, x in enumerate(newsgroups_train.target) if x == 7]
indices8 = [i for i, x in enumerate(newsgroups_train.target) if x == 8]
indices9 = [i for i, x in enumerate(newsgroups_train.target) if x == 9]
indices10 = [i for i, x in enumerate(newsgroups_train.target) if x == 10]
indices11 = [i for i, x in enumerate(newsgroups_train.target) if x ==11 ]
indices12 = [i for i, x in enumerate(newsgroups_train.target) if x ==12 ]
indices13 = [i for i, x in enumerate(newsgroups_train.target) if x ==13 ]
indices14 = [i for i, x in enumerate(newsgroups_train.target) if x ==14 ]
indices15 = [i for i, x in enumerate(newsgroups_train.target) if x ==15 ]
indices16 = [i for i, x in enumerate(newsgroups_train.target) if x ==16 ]
indices17 = [i for i, x in enumerate(newsgroups_train.target) if x ==17 ]
indices18 = [i for i, x in enumerate(newsgroups_train.target) if x ==18 ]
indices19 = [i for i, x in enumerate(newsgroups_train.target) if x ==19 ]

indices_one = [indices0,indices1,indices2,indices3,indices4,indices5,indices6,indices7,indices8,indices9,indices10,indices11,indices12,indices13,indices14,indices15,indices16,indices17,indices18,indices1]

z = np.zeros([20,1000,4096])
counts=np.zeros(20)
for i in range(20):
	for j in range(len(indices_one[i])):
		z[i,int(counts[i])] = vectors1[indices_one[i][j]]
		counts[i]+=1

indices0 = [i for i, x in enumerate(newsgroups_test.target) if x ==0 ]
indices1 = [i for i, x in enumerate(newsgroups_test.target) if x == 1]
indices2 = [i for i, x in enumerate(newsgroups_test.target) if x == 2]
indices3 = [i for i, x in enumerate(newsgroups_test.target) if x == 3]
indices4 = [i for i, x in enumerate(newsgroups_test.target) if x == 4]
indices5 = [i for i, x in enumerate(newsgroups_test.target) if x == 5]
indices6 = [i for i, x in enumerate(newsgroups_test.target) if x == 6]
indices7 = [i for i, x in enumerate(newsgroups_test.target) if x == 7]
indices8 = [i for i, x in enumerate(newsgroups_test.target) if x == 8]
indices9 = [i for i, x in enumerate(newsgroups_test.target) if x == 9]
indices10 = [i for i, x in enumerate(newsgroups_test.target) if x == 10]
indices11 = [i for i, x in enumerate(newsgroups_test.target) if x ==11 ]
indices12 = [i for i, x in enumerate(newsgroups_test.target) if x ==12 ]
indices13 = [i for i, x in enumerate(newsgroups_test.target) if x ==13 ]
indices14 = [i for i, x in enumerate(newsgroups_test.target) if x ==14 ]
indices15 = [i for i, x in enumerate(newsgroups_test.target) if x ==15 ]
indices16 = [i for i, x in enumerate(newsgroups_test.target) if x ==16 ]
indices17 = [i for i, x in enumerate(newsgroups_test.target) if x ==17 ]
indices18 = [i for i, x in enumerate(newsgroups_test.target) if x ==18 ]
indices19 = [i for i, x in enumerate(newsgroups_test.target) if x ==19 ]

indices_two = [indices0,indices1,indices2,indices3,indices4,indices5,indices6,indices7,indices8,indices9,indices10,indices11,indices12,indices13,indices14,indices15,indices16,indices17,indices18,indices1]

for i in range(20):
	for j in range(len(indices_two[i])):
		z[i,int(counts[i])] = vectors2[indices_two[i][j]]
		counts[i]+=1

np.save('tfidf.npy', z)
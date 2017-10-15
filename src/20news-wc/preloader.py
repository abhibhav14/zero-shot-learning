from __future__ import print_function
import nltk
import numpy as np

import argparse
import string
import sys
import os
import operator
import time

from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))


parser= argparse.ArgumentParser("process data into word count vector")
parser.add_argument('--source',dest='sdir', type=str, default="Source file directory",
	help="Source directory of input files")
parser.add_argument('--vocab_size', dest='vocab_size', type=int, default=10000,
	help="Set the vocabulary size for processing")
parser.add_argument('--target', dest='target_dir', type=str, default="/new_data/gpu/prannay/20news",
	help="Directory for storing processed data")
args = parser.parse_args()

tokenizer = TreebankWordTokenizer()
stemmer = LancasterStemmer()

skip_chars = "@#$%&*()$!~"

def vocabfunc(x):
	if x in vocab :
		vocab[x] += 1
	else :
		vocab[x] = 0
	return x

start_time = time.time()
vocab = {}

classes = [f for f in os.listdir(args.sdir)]
print(classes)
for f in classes :
	files = [a for a in os.listdir(os.path.join(args.sdir, f))]
	for fil in files:
		with open(os.path.join(args.sdir + "/" + f, fil)) as fopen :
			text = filter(lambda x : (x in string.printable) and (not x in skip_chars),' '.join(fopen.readlines())) # read the entire file as a single line
			text = text.lower()
			text = tokenizer.tokenize(text)
			text = [stemmer.stem(word) for word in text if (not word in stopwords) and (len(word) > 1)]
			map(lambda x : vocabfunc(x), text)
			print("Completed : {} {}".format(f, fil), end='\r')
print()
print("Creating vocab with size : {}".format(len(vocab)))

vocab = [i[0] for i in sorted(vocab.items(), key=operator.itemgetter(1))][-args.vocab_size:]
print(len(vocab))
print(vocab[:10])

token_dict = {}
for word in vocab:
	token_dict[word] = len(token_dict)

docidx = [0] * 20
classidx = 0
for f in classes :
	classidx +=1
	files = [a for a in os.listdir(os.path.join(args.sdir, f))]
	for fil in files:
		with open(os.path.join(args.sdir + "/" + f, fil)) as fopen :
			count = [0] * (args.vocab_size+1)
			text = filter(lambda x : (x in string.printable) and (not x in skip_chars),' '.join(fopen.readlines())) # read the entire file as a single line
			text = text.lower()
			text = tokenizer.tokenize(text)
			text = [stemmer.stem(word) for word in text if (not word in stopwords) and (len(word) > 1)]
			start_t = time.time()
			for word in text :
				if word in vocab :
					count[token_dict[word]] += 1
				else :
					count[-1] += 1
			count_mat = np.array(count).astype(np.float32)
			count_mat = count_mat / np.sqrt(np.sum(np.square(count_mat)))
			np.save("{}/{}_{}".format(args.target_dir,classidx,docidx[classidx-1]),
				count_mat)
			docidx[classidx - 1] += 1
			print("Completed : {} {} in {}".format(f, fil, time.time() - start_t), end='\r')
print()
print("Total time : {}".format(time.time() - start_time))

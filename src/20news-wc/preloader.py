import nltk
import argparse
import string
import sys
import os
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
parser= argparse.ArgumentParser("process data into word count vector")
parser.add_argument('--source',dest='sdir', type=str, default="Source file directory",
	help="Source directory of input files")
args = parser.parse_args()

tokenizer = TreebankWordTokenizer()
stemmer = LancasterStemmer()

skip_chars = "@#$%&*()$!~"

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
			print(text[:5])

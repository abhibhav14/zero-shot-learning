from __future__ import print_function
import numpy as np
import argparse
import os


parser= argparse.ArgumentParser("process data into word count vector")
parser.add_argument('--source',dest='sdir', type=str, default="Source file directory",
	help="Source directory of input files")
parser.add_argument('--vocab_size', dest='vocab_size', type=int, default=10000,
	help="Set the vocabulary size for processing")
parser.add_argument('--num_class', dest="num_class", type=int, default=10,
	help="Set the number of classes")
parser.add_argument('--num_dimension', dest="dimension", type=int, default=10,
	help="Set the number of classes")
parser.add_argument('--target', dest='target_dir', type=str, default="/new_data/gpu/prannay/20news",
	help="Directory for storing processed data")
args = parser.parse_args()

train_file_list = [f for f in os.listdir(args.sdir) ]
class_list = {}
for i in range(args.num_class):
	class_list["class_%d"%(i)] = list()
for fil in train_file_list[:1000]:
	classnum = int(fil.split("_")[0]) - 1
	class_list["class_%d"%(classnum)].append(np.load(os.path.join(args.sdir,fil)))
	print("Done with {}".format(fil), end="\r")
print()
class_means = np.zeros([args.num_class, args.dimension])
class_kappa = np.zeros([args.num_class, args.dimension])

for i in range(args.num_class):
	t = np.array(class_list["class_%d"%(i)], copy=True)
	print()
	print(t.shape)
	class_means[i] = np.sum(t,axis=0)
	class_means[i] = class_means[i] / np.sqrt(np.sum(np.square(class_means[i])))
	map(lambda x : x - class_means[i], t)
	class_cov[i] = np.var(t,axis=0)
	print("Done with {}".format(i), end="\r")

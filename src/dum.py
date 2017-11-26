from __future__ import print_function
import numpy as np

def loadWordVectors():
    print ("Loading Glove Model")
    f = open("/new_data/gpu/prannay/glove.6B.100d.txt",'r')
    print("Loaded model, now understanding etc..")
    model = {}
    count = 0
    for line in f:
        print(count, end="\r")
        count+= 1
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

vec = loadWordVectors()

attr = np.concatenate((vec['comp'],vec['graphics'], vec['unk']))
attr = np.vstack((attr,np.concatenate((vec['computer'],vec['windows'],vec['microsoft']))))
attr = np.vstack((attr,np.concatenate((vec['computer'],vec['hardware'],vec['ibm']))))
attr = np.vstack((attr,np.concatenate((vec['computer'],vec['hardware'],vec['macintosh']))))
attr = np.vstack((attr,np.concatenate((vec['computer'],vec['windows'],vec['software']))))
attr = np.vstack((attr,np.concatenate((vec['record'],vec['automobiles'],vec['transport']))))
attr = np.vstack((attr,np.concatenate((vec['record'],vec['motorcycles'],vec['transport']))))
attr = np.vstack((attr,np.concatenate((vec['record'],vec['sport'],vec['baseball']))))
attr = np.vstack((attr,np.concatenate((vec['record'],vec['sport'],vec['hockey']))))
attr = np.vstack((attr,np.concatenate((vec['science'],vec['cryptography'],vec['cipher']))))
attr = np.vstack((attr,np.concatenate((vec['science'],vec['electronics'],vec['hardware']))))
attr = np.vstack((attr,np.concatenate((vec['science'],vec['medicine'],vec['pfizer']))))
attr = np.vstack((attr,np.concatenate((vec['science'],vec['space'],vec['nasa']))))
attr = np.vstack((attr,np.concatenate((vec['sale'],vec['market'],vec['money']))))
attr = np.vstack((attr,np.concatenate((vec['talk'],vec['politics'],vec['government']))))
attr = np.vstack((attr,np.concatenate((vec['talk'],vec['politics'],vec['guns']))))
attr = np.vstack((attr,np.concatenate((vec['talk'],vec['politics'],vec['mideast']))))
attr = np.vstack((attr,np.concatenate((vec['talk'],vec['religion'],vec['misc']))))
attr = np.vstack((attr,np.concatenate((vec['alternative'],vec['atheism'],vec['religion']))))
attr = np.vstack((attr,np.concatenate((vec['society'],vec['religion'],vec['christian']))))
np.save('model.npy', attr)
print(attr.shape)

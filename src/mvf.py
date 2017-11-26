"""
Reimplimentation of the model described by us

In this model, we consider that each class is a von Mises Fischer,
whose parameters are linear functions of the class attribute
vector.

Authors: Abhibhav, Prannay, Soumye
"""
from __future__ import print_function
import numpy as np
import scipy.linalg as sp
import scipy.stats as stats
from sklearn.linear_model import Ridge

factorial = [0]
np.set_printoptions(threshold=np.nan)

def inference(unseenList, muOut, kappaOut, testcase):
  pos = np.zeros(50) - 100000000
  for i in unseenList : 
    pos[i] = logmvfpdf(testcase, muOut[i], kappaOut[i])
  ex = np.exp(pos - np.max(pos))
  ex = ex / ex.sum()
  return np.argsort(ex)[-5:], np.sort(ex)[-5:]

def fact(i):
  if i >= len(factorial):
    j = len(factorial)
    while(j <= i):
      j += 1
      factorial.append(factorial[-1] + np.log(j-1))
  return factorial[i]

def calcI(p,k):
  sumval = 0
  prod = p*np.log(k / 2.)
  for i in range(100):
    a = fact(int(p+i+1)) + fact(i)
    sumval += (prod - a)
    prod += 2*np.log(k/2.)
  return sumval

def logmvfpdf(t, m, k):
  I_pval = calcI((10001/2.) - 1, k)
  cp = ((10001 / 2.) -1)*np.log(k) - ((10001 / 2)*np.log(2*np.pi) + I_pval)
  return k*np.dot(m,t) - cp

def createModel(data=None,
                classFeatures=None,
                featureDimension=10001,
                classFeatureDimension=300,
                meanRegularizer=0, meanVariance = 10,
                varRegularizer=0, varVariance = 10,
                modelPath="./zslPoint"):
    """
    Function that loads and does point based inference for vmF distribution
    args:
        data: file path to the data
        classFeatures: file path to class features
        featureDimension: dimensionality of features
        classFeatureDimension: dimensionality of class feature vector
        meanRegularizer: hyperparameter for l2 loss for mean
        varRegularizer: hyperparameter for l2 loss for variance
        model: file name for model
    returns:
        None
    """
    D = np.load("../../20news/data/features.npy")
    A = np.load("../../20news/data/classFeat.npy")
    counts =  np.zeros([20], dtype=np.int16)
    for i in range(20):
      for j in range(1000):
        if np.mean(D[i,j]) == 0:
          break
      counts[i] = j
    print("Data Loaded")
    seenList = [4,5,10,12,14] 
    seenclassData = list()
    seenclassfeatures = list()
    seenClassList = list()
    unseenclassfeatures = list()
    unseenList = list()

    for i in range(20):
        if not i in seenList:
            seenclassData.append(D[i][:counts[i]])
            seenclassfeatures.append(A[i])
            seenClassList.append(i)
        else:
            unseenclassfeatures.append(A[i])
            unseenList.append(i)
    testD = np.copy(D)
    D = None
    empMu = np.zeros([len(seenclassData), featureDimension])
    empKappa = np.zeros([len(seenclassData)])
    for i in range(len(seenclassData)):
      count = counts[seenClassList[i]]
      empMean = np.mean(seenclassData[i], axis=0)
      r = count * empMean
      if not np.sqrt(np.sum(np.square(r))) < 1e-10:
        empMu[i]  = r / np.sqrt(np.sum(np.square(r)))
      else : 
        empMu[i] = np.zeros_like(r)
      print(np.mean(empMu[i]))
      Rbar = np.sqrt(np.sum(np.square(r))) / count + 1e-8
      empKappa[i] = (Rbar* featureDimension - Rbar ** 3) / (1 - Rbar ** 2)

    empKappa = np.log(empKappa)

    modelMean = Ridge(alpha = 3200)
    modelKappa = Ridge(alpha = 3500)

    modelMean.fit(seenclassfeatures, empMu)
    modelKappa.fit(seenclassfeatures, empKappa)

    meanOut = modelMean.predict(A)
    kappaOut = np.exp(modelKappa.predict(A))
    print("Model learnt")

#    testD = np.load("../../20news/data/test_feat.npy")
    for i in unseenList:
      print(kappaOut[i])
      for j in range(min(100,counts[i])):
        pred, vals = inference(unseenList + seenClassList, meanOut, kappaOut, testD[i,j])
        writevar = "{} : {} : {}\n".format(i, ' '.join(map(lambda x : str(x), pred)[::-1]), ' '.join(map(lambda x : str(x), vals)[::-1]))
        print(writevar, end="\r")
        with open("out_mvf_point.txt", mode="a") as f: f.write(writevar)
      print()

createModel()

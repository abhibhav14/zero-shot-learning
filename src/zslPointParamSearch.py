"""
Reimplimentation of the model described in Verma and Rai.

In this model, we consider that each class is a Gaussian,
whose parameters are linear functions of the class attribute
vector.

The high level steps are as follows:
    1. Construct emperical estimates of the parameters
        for the seen classes.
    2. Learn a function that can map the class attributes
        to these parameters.
    3. Apply this function to the class attributes of the
        unseen classes to get parameters for its generative
        model.
    4. During training, predict classes based on how likely
        is the data to be generated from each class.

Authors: Abhibhav, Prannay, Soumye
"""

from __future__ import print_function
import numpy as np
import scipy.linalg as sp
import scipy.stats as stats
from sklearn.linear_model import Ridge

def inference (unseenList, muOut,  empOut, point):
  pos = np.zeros(50) - 1e7
  for i in unseenList : 
    pos[i] = np.sum([stats.norm.logpdf(point[j], muOut[i][j], empOut[i][j]) for j in range(4096)])
  ex = np.exp(pos - np.max(pos))
  ex = ex/ex.sum()
  return np.argsort(pos)[-5:]

def createModel(param1, param2,
                data=None,
                classFeatures=None,
                numSeenClasses=40,
                numUnseenClasses=10,
                featureDimension=4096,
                classFeatureDimension=85,
                meanRegularizer=0,
                varRegularizer=0,
                modelPath="./zslPoint"):
    """
    Function that creates and saves a model
    args:
        data: file path to the data
        classFeatures: file path to class features
        numSeenClasses: number of seen classes
        numUnSeenClasses: number of unseen classes
        featureDimension: dimensionality of features
        classFeatureDimension: dimensionality of class feature vector
        meanRegularizer: hyperparameter for l2 loss for mean
        varRegularizer: hyperparameter for l2 loss for variance
        model: file name for model
    returns:
        None
    """

    # TODO load data based on the format
    # For now, assume that the matrix D has the data
    # D[i][j] is the feature vector of the jth data point
    # of the ith class

    # In this work, Verma and Rai took simple empirical means and variances
    # and used those as parameters for the generative model. This is
    # precisely what we want to try and improve first.
    # These vectors have shape dimension x numClasses

    D = np.load("../../Awa_zeroshot/awadata/train_feat.npy")
    A = np.load("../../Awa_zeroshot/awadata/classFeat.npy")
    counts = np.load("../../Awa_zeroshot/awadata/count_vector.npy").astype(int)

    seenclassData = list()
    seenclassfeatures = list()
    seenClassList = list()
    unseenclassfeatures = list()
    unseenList = list()
  
    for i in range(50):
      if counts[i] != 0:
        seenclassData.append(D[i][:counts[i]])
        seenclassfeatures.append(A[i])
        seenClassList.append(i)
      else :
        unseenclassfeatures.append(A[i])
        unseenList.append(i)

    D = None
    empMu = np.zeros([len(seenclassData), featureDimension])
    empVar = np.zeros_like(empMu) 

    # TODO load the class features
    # A will be an array of shape numClassFeatures x numSeenClasses

    # Calculate mappings from class attributes
    # to the generative model params

    for i in range(len(seenclassData)):
      count = counts[seenClassList[i]]
      empMu[i] = np.mean(seenclassData[i], axis=0)
      empVar[i] = np.var(seenclassData[i], axis=0) + 1e-6
    empVar = np.log(empVar)
  
    modelMu = Ridge(alpha = param1)
    modelEmpV = Ridge(alpha = param2)

    modelMu.fit(seenclassfeatures, empMu)
    modelEmpV.fit(seenclassfeatures, empVar)

    muOut = modelMu.predict(A)
    varOut = np.exp(modelEmpV.predict(A))


    testD = np.load("../../Awa_zeroshot/awadata/test_feat.npy")
    countsTest = np.load("../../Awa_zeroshot/awadata/countsTest.npy").astype(int)
    correct = 0
    for i in unseenList :
      np.random.permutation(testD[i])
      for j in range(min(countsTest[i],20)):
        pred = inference(unseenList, muOut, varOut, testD[i][j])
        if pred[-1] == i:
          correct += 1

    return correct

best = None
prevBest = None

param = [30000,100000,300000,1000000,3000000,10000000]
for i in param:
  for j in param:
    print("starting : {}".format(j+1))
    pred = createModel(param1 = i, param2 = j)
    if prevBest is None or pred > prevBest:
      prevBest = pred
      best = (i, j)
    print("Ending : {}".format(j+1))

print(best)

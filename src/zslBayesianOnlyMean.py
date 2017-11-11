"""
Reimplimentation of the model described in Verma and Rai.

In this model, we consider that each class is a Gaussian,
whose parameters are linear functions of the class attribute
vector.

The high level steps are as follows:
    1. Construct emperical estimates of the parameters
        for the seen class.
    2. Learn a function that can map the class attributes
        to these parameters.
    3. Apply this function to the class attributes of the
        unseen class to get parameters for its generative
        model.
    4. During training, predict class based on how likely
        is the data to be generated from each class.

Authors: Abhibhav, Prannay, Soumye
"""
from __future__ import print_function
import numpy as np
import scipy.linalg as sp
import scipy.stats as stats
from sklearn.linear_model import Ridge

np.set_printoptions(threshold=np.nan)

def createModel(data=None,
                classFeatures=None,
                featureDimension=4096,
                classFeatureDimension=85,
                meanRegularizer=0, meanVariance = 10,
                varRegularizer=0, varVariance = 10,
                modelPath="./zslPoint"):
    """
    Function that creates and saves a model
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
    D = np.load("../../Awa_zeroshot/awadata/train_feat.npy")
    A = np.load("../../Awa_zeroshot/awadata/classFeat.npy")
    counts = np.load("../../Awa_zeroshot/awadata/count_vector.npy")
    counts = counts.astype(int)
    print("Data Loaded")

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
        else:
            unseenclassfeatures.append(A[i])
            unseenList.append(i)
    D = None

    # Hyperparameters
    pmu = np.zeros((len(seenclassData), featureDimension))
    psig = np.ones((len(seenclassData), featureDimension))
    empVar = np.zeros_like(pmu)
    # plambda = np.ones((len(seenclassData), featureDimension))
    # palpha = np.ones((len(seenclassData), featureDimension)) * 0.001
    # pbeta = np.ones((len(seenclassData), featureDimension)) * 0.001

    # All of this stuff should be technically done via vector calcs
    # but doing individually is easier

    for i in range(len(seenclassData)):
        count = counts[seenClassList[i]]
        empMean = np.mean(seenclassData[i], axis=0)
        empVar[i] = np.var(seenclassData[i], axis=0) + 1e-6
        pmu[i] = (empVar[i] * pmu[i] + count * empMean * psig[i]) / (count * psig[i] + empVar[i])
        psig[i] = 1 / (count/empVar[i] + 1/psig[i])



    # We pass the lambda, alpha and beta through a log function for positivity
    psig = np.log(psig)
    empVar = np.log(empVar)

    # Calulate the lin reg outputs
    # wmu = sp.lstsq(seenclassfeatures, pmu)[0]
    # wlambda = sp.lstsq(seenclassfeatures, plambda)[0]
    # walpha = sp.lstsq(seenclassfeatures, palpha)[0]
    # wbeta = sp.lstsq(seenclassfeatures, pbeta)[0]

    modelMu = Ridge(alpha = 325000)
    modelSig = Ridge(alpha = 32500)
    modelEmpV = Ridge(alpha = 32500)

    modelMu.fit(seenclassfeatures, pmu)
    modelSig.fit(seenclassfeatures, psig)
    modelEmpV.fit(seenclassfeatures, empVar)


    # Calculate the params for all classes
    muOut = modelMu.predict(A)
    sigOut = np.exp(modelSig.predict(A))
    empSigOut = np.exp(modelEmpV.predict(A))

    print("Model Learnt")

    testD = np.load("../../Awa_zeroshot/awadata/test_feat.npy")
    countsTest = np.load("../../Awa_zeroshot/awadata/countsTest.npy")
    countsTest = countsTest.astype(int)
    print("Inferring")
    for i in unseenList:
        for j in range(countsTest[5]):
            pred,vals = inference(unseenList, muOut, sigOut, empSigOut, testD[i][j])
            writevar = "{} : {} : {}\n".format(i, ' '.join(map(lambda x : str(x),pred)[::-1]), ' '.join(map(lambda x : str(x),vals)[::-1]))
            print(writevar, end='\r')
            with open("out_mean.txt", mode="a") as f: f.write(writevar)
        print()

    

def inference(unseenList, muOut, sigOut, empSigOut, point):
    pos = np.zeros(50) - 10000000
    for i in unseenList:
        pos[i] = np.sum([stats.multivariate_normal.logpdf(point[j], muOut[i][j], sigOut[i][j] + empSigOut[i][j]) for j in range(4096)])
    ex = np.exp(pos - np.max(pos))
    ex = ex / ex.sum()
    return np.argsort(pos)[-5:], np.sort(ex)[-5:]

createModel()

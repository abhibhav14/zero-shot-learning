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
    print(counts)

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
    # pmu = np.zeros((len(seenclassData), featureDimension))
    # psig = np.ones((len(seenclassData), featureDimension))
    empMean = np.zeros((len(seenclassData), featureDimension))
    empVar = np.zeros_like(empMean)
    palpha = np.ones_like(empMean) * 2
    pbeta = np.zeros_like(empMean)

    # plambda = np.ones((len(seenclassData), featureDimension))
    # palpha = np.ones((len(seenclassData), featureDimension)) * 0.001
    # pbeta = np.ones((len(seenclassData), featureDimension)) * 0.001

    # All of this stuff should be technically done via vector calcs
    # but doing individually is easier

    for i in range(len(seenclassData)):
        count = counts[seenClassList[i]]
        empMean[i] = np.mean(seenclassData[i], axis=0)
        empVar[i] = np.var(seenclassData[i], axis=0) + 1e-6
        # pmu[i] = (empVar[i] * pmu[i] + counts[i] * empMean * psig[i]) / (counts[i] * psig[i] + empVar[i])
        # psig[i] = 1 / (counts[i]/empVar[i] + 1/psig[i])
        palpha[i] = palpha[i] + count / 2
        pbeta[i] = pbeta[i] + 0.5 * empVar[i] * count
        print(min(pbeta[i]))


    # We pass the lambda, alpha and beta through a log function for positivity
    palpha = np.log(palpha)
    pbeta = np.log(pbeta)

    # Calulate the lin reg outputs
    # wmu = sp.lstsq(seenclassfeatures, pmu)[0]
    # wlambda = sp.lstsq(seenclassfeatures, plambda)[0]
    # walpha = sp.lstsq(seenclassfeatures, palpha)[0]
    # wbeta = sp.lstsq(seenclassfeatures, pbeta)[0]

    modelMu = Ridge(alpha = 325000)
    modelpal = Ridge(alpha = 32500)
    modelpbe = Ridge(alpha = 32500)

    modelMu.fit(seenclassfeatures, empMean)
    modelpal.fit(seenclassfeatures, palpha)
    modelpbe.fit(seenclassfeatures, pbeta)


    # Calculate the params for all classes
    muOut = modelMu.predict(A)
    palOut = np.exp(modelpal.predict(A))
    pbeOut = np.exp(modelpbe.predict(A))
    print("Reached")

    testD = np.load("../../Awa_zeroshot/awadata/test_feat.npy")
    countsTest = np.load("../../Awa_zeroshot/awadata/countsTest.npy")
    countsTest = countsTest.astype(int)
    print("Infering")
    for j in range(countsTest[5]):
        for i in unseenList:
            print(i, inference(unseenList, muOut, palOut, pbeOut, testD[i][j]), j, " out of ", countsTest[i], " samples from this class.")
        print()

    return

def inference(unseenList, muOut, palOut, pbeOut, point):
    pos = np.zeros(50) - 10000000
    for i in unseenList:
        pos[i] = np.sum([stats.t.logpdf(point[j], 2 * palOut[i][j], muOut[i][j], pbeOut[i][j] / palOut[i][j]) for j in range(4096)])
    print(pos)
    return np.argmax(pos)

createModel()

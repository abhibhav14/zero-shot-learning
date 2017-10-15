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
    unseenclassfeatures = list()

    for i in range(50):
        if counts[i] != 0:
            seenclassData.append(D[i][:counts[i]])
            seenclassfeatures.append(A[i])
        else:
            unseenclassfeatures.append(A[i])
    D = None

    # Hyperparameters
    pmu = np.zeros((len(seenclassData), featureDimension))
    plambda = np.ones((len(seenclassData), featureDimension))
    palpha = np.ones((len(seenclassData), featureDimension))
    pbeta = np.ones((len(seenclassData), featureDimension)) * 0.01

    # All of this stuff should be technically done via vector calcs
    # but doing individually is easier

    for i in range(len(seenclassData)):
        empMean = np.mean(seenclassData[i], axis=0)
        empVar = np.var(seenclassData[i], axis=0)
        palpha[i] += (counts[i] / 2)
        pbeta[i] += 0.5 * (counts[i] * empVar + ((plambda[i] * counts[i] * (empMean - pmu[i]) * (empMean - pmu[i])) / (plambda[i] + counts[i])))
        pmu[i] = (plambda[i] * pmu[i] + counts[i] * empMean) / (plambda[i] + counts[i])
        plambda[i] += counts[i]


    # We pass the lambda, alpha and beta through a log function for positivity
    palpha = np.log(palpha)
    pbeta = np.log(pbeta)
    plambda = np.log(plambda)

    # Calulate the lin reg outputs
    wmu = sp.lstsq(seenclassfeatures, pmu)[0]
    wlambda = sp.lstsq(seenclassfeatures, plambda)[0]
    walpha = sp.lstsq(seenclassfeatures, palpha)[0]
    wbeta = sp.lstsq(seenclassfeatures, pbeta)[0]

    # Calculate the params for all classes
    muOut = np.matmul(A, wmu)
    lambdaOut = np.exp(np.matmul(A, wlambda))
    alphaOut = np.exp(np.matmul(A, walpha))
    betaOut = np.exp(np.matmul(A, wbeta))
    print(betaOut.shape)

    testD = np.load("../../Awa_zeroshot/awadata/test_feat.npy")
    countsTest = np.load("../../Awa_zeroshot/awadata/countsTest.npy")
    countsTest = countsTest.astype(int)
    print("here")
    print(inference(muOut, lambdaOut, alphaOut, betaOut, testD[0][1]))

    return

def inference(muOut, lambdaOut, alphaOut, betaOut, point):
    pos = np.zeros(50)
    for i in range(50):
        pos[i] = np.sum(stats.t.logpdf(point, alphaOut[i] * 2, muOut[i], (betaOut[i] * (lambdaOut[i] + 1)) / (alphaOut[i] * lambdaOut[i])))
    return np.argmax(pos)

createModel()

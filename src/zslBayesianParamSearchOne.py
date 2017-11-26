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
import time

np.set_printoptions(threshold=np.nan)

def createModel(regAlpha, data=None,
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

    seenclassData = list()
    seenclassfeatures = list()
    unseenclassfeatures = list()
    seenClassList = list()
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
    plambda = np.ones((len(seenclassData), featureDimension))
    palpha = np.ones((len(seenclassData), featureDimension)) * 2
    pbeta = np.ones((len(seenclassData), featureDimension)) * 0

    # All of this stuff should be technically done via vector calcs
    # but doing individually is easier

    for i in range(len(seenclassData)):
        count = counts[seenClassList[i]]
        empMean = np.mean(seenclassData[i], axis=0)
        empVar = np.var(seenclassData[i], axis=0)
        palpha[i] += (count / 2)
        pbeta[i] += 0.5 * (count * empVar + ((plambda[i] * count * (empMean - pmu[i]) * (empMean - pmu[i])) / (plambda[i] + count)))
        pmu[i] = (plambda[i] * pmu[i] + count * empMean) / (plambda[i] + count)
        # plambda[i] += counts[i]
        # print(max(pmu[i]))
        # print(min(pmu[i]))
        # print(len(seenclassData[i]))


    # We pass the lambda, alpha and beta through a log function for positivity
    palpha = np.log(palpha)
    pbeta = np.log(pbeta + 1e-9)
    plambda = np.log(plambda)

    # Calulate the lin reg outputs
    # wmu = sp.lstsq(seenclassfeatures, pmu)[0]
    # wlambda = sp.lstsq(seenclassfeatures, plambda)[0]
    # walpha = sp.lstsq(seenclassfeatures, palpha)[0]
    # wbeta = sp.lstsq(seenclassfeatures, pbeta)[0]

    modelMu = Ridge(alpha=regAlpha)
    modelLambda = Ridge(alpha=regAlpha)
    modelAlph = Ridge(alpha=regAlpha)
    modelBet = Ridge(alpha=regAlpha)


    modelMu.fit(seenclassfeatures, pmu)
    modelLambda.fit(seenclassfeatures, plambda)
    modelAlph.fit(seenclassfeatures, palpha)
    modelBet.fit(seenclassfeatures, pbeta)

    # Calculate the params for all classes
    muOut = modelMu.predict(A)
    lambdaOut = np.exp(modelLambda.predict(A))
    alphaOut = np.exp(modelAlph.predict(A))
    betaOut = np.exp(modelBet.predict(A))

    testD = np.load("../../Awa_zeroshot/awadata/test_feat.npy")
    countsTest = np.load("../../Awa_zeroshot/awadata/countsTest.npy")
    countsTest = countsTest.astype(int)
    countAcc = 0
    start_time = time.time()
    for i in unseenList:
        for j in np.random.permutation(countsTest[i])[:200]:
            pred,vals = inference(unseenList, muOut, lambdaOut, alphaOut, betaOut, testD[i][j])
            if i in pred:
    #          print("{} : {}".format(i+1, j+1))
              countAcc += 1
            # writevar = "{} : {} : {}\n".format(i, ' '.join(map(lambda x : str(x),pred)[::-1]), ' '.join(map(lambda x : str(x),vals)[::-1]))
            # print(writevar, end='\r')
            #with open("out_full.txt", mode="a") as f: f.write(writevar)
        #print()
    
    return countAcc

def inference(unseenList, muOut, lambdaOut, alphaOut, betaOut, point, ):
    pos = np.zeros(50) - 10000000
    lambdaOut = np.ones_like(lambdaOut)
    alphaOut = np.ones_like(alphaOut)
    for i in unseenList:
        pos[i] = np.sum(stats.t.logpdf(point, alphaOut[i] * 2, muOut[i], (betaOut[i] * (lambdaOut[i] + 1)) / (alphaOut[i] * lambdaOut[i])))
    ex = np.exp(pos - np.max(pos))
    ex = ex / ex.sum()
    return np.argsort(pos)[-3:], np.sort(ex)[-5:]

alphas = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 1000000, 3000000]
for i in range(len(alphas)):
    print("Starting for {} with hyper-params : {}".format(i+1,alphas[i]))
    start_time = time.time()
    c = createModel(alphas[i])
    print("Ending for {} with hyper-params : {} in time: {}".format(i+1,alphas[i], time.time() - start_time))
    print(alphas[i], c)

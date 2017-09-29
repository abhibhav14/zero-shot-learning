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

import numpy as np

def createModel(data=None,
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

    D = np.load("../../awa/features.npy")
    D = D[10:]
    A = np.load("../../awa/classes.npy")
    A = np.transpose(A[10:])
    empirical_means = np.transpose(np.mean(D, axis=1))
    log_empirical_vars = np.log(np.transpose(np.var(D, axis=1)))

    # TODO load the class features
    # A will be an array of shape numClassFeatures x numSeenClasses

    # Calculate mappings from class attributes
    # to the generative model params

    W_mean = np.matmul(A, np.transpose(A))
    W_mean += meanRegularizer * np.eye(classFeatureDimension)
    W_mean = np.linalg.inv(W_mean)
    W_mean = np.matmul(np.transpose(A), W_mean)
    W_mean = np.matmul(empirical_means, W_mean)

    W_var = np.matmul(A, np.transpose(A))
    W_var += varRegularizer * np.eye(classFeatureDimension)
    W_var = np.linalg.inv(W_var)
    W_var = np.matmul(np.transpose(A), W_var)
    W_var = np.matmul(log_empirical_vars, W_var)

    # TODO matrix A now contains all class features
    A = np.transpose(np.load("../../awa/classes.npy"))
    means = np.matmul(W_mean, A)
    vars = np.exp(np.matmul(W_var, A))

    np.save("model", np.stack((means, vars)))

    return
createModel()

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
				meanRegularizer=0, meanVariance = 10,
				varRegularizer=0, varVariance = 10,
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
	D = np.load("../../awa/features.npy")
	D = D[10:]
	A = np.load("../../awa/classes.npy")
	A = np.transpose(A[10:])
	empirical_means = np.transpose(np.mean(D, axis=1))
	log_empirical_vars = np.log(np.transpose(np.var(D, axis=1)))

	W_mean = meanVariance * np.matmul(A, np.transpose(A))
	W_mean += meanRegularizer * np.eye(classFeatureDimension)
	W_mean = np.linalg.inv(W_mean)
	mu_mean = meanVariance * np.matmul(np.transpose(A), W_mean)
	mu_mean = np.matmul(empirical_means, mu_mean)
	W_var = varVariance * np.matmul(A, np.transpose(A))
	W_var += varRegularizer * np.eye(classFeatureDimension)
	W_var = np.linalg.inv(W_var)
	mu_var = varVariance * np.matmul(np.transpose(A), W_var)
	mu_var = np.matmul(log_empirical_vars, mu_var)

	A = np.transpose(np.load("../../awa/classes.npy"))
	means = np.matmul(mu_mean, A)
	mean_variance = np.zeros(means.shape)
	for i in range(A.shape[1]):
		vari = meanVariance + np.matmul(np.transpose(A[:,i]), np.matmul(W_mean, A[:,i]))
		mean_variance[i] *= vari
	""" variance = meanVariance * np.ones(means.shape) + np.matmul(np.transpose(A),np.matmul(W_mean,A)) """
	vars = np.exp(np.matmul(mu_var, A))
	var_variance = np.zeros(vars.shape)
	for i in range(A.shape[1]):
		vari = varVariance + np.matmul(np.transpose(A[:,i]), np.matmul(W_var, A[:,i]))
		var_variance[i] *= vari
	np.save("model_bayesian", np.stack((means, vars, mean_variance, var_variance)))

	return
createModel()

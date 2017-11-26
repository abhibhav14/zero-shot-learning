import numpy as np
import scipy as sc
import scipy.stats as scstats
def rW(n, kappa, m):
	dim = m -1
	b = dim / (np.sqrt(4*kappa*kappa + dim*dim) + 2*kappa)
	x = (1 - b) / (1 + b)
	c = kappa * x + dim*np.log(1 - x*x)
	
	y = []
	done = False
	for i in range(0,n):
		while not done :
			z = sc.stats.beta.rvs(dim/2, dim/2)
			w = (1 - (1+b)*z) / (1 - (1 - b)*z)
			u = scstats.uniform.rvs()
			if kappa*w + dim*np.log(1 - x*w) - c >= np.log(u):
				done = True
		y.append(w)
	return y

def rvMF(n, theta):
	dim = len(theta)
	kappa = np.linalg.norm(theta)
	mu = theta / kappa
	result = []
	for sample in range(n):
		w = rW(n, kappa, dim)
		v = np.random.randn(dim)
		v = v / np.linalg.norm(v)
		print(v)
		result.append(np.sqrt(1 - w**2)*v + w*mu)
	return result

n = 10
kappa = 10000
direction = np.array([1,-1,1])
direction = direction / np.linalg.norm(direction)

res_sampling = rvMF(n, kappa*direction)
print(np.array(res_sampling).shape)

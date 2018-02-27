
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from IPython import embed
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata

np.random.seed(1)

L=100

Xg,Yg = np.meshgrid(range(L),range(L))

def visualize_map(map,title,probed_points=None):
	map=map.reshape(L,-1)
	plt.clf()
	plt.imshow(map,origin='lower')
	plt.title(title)
	plt.colorbar()
	if not (probed_points is None):
		plt.scatter(probed_points[:,0],probed_points[:,1])
	plt.xlim((0,100))
	plt.ylim((0,100))
	# plt.tight_layout()
	# plt.show()
	plt.pause(0.01)

def evaluateStiffness(grid, groundTruth, X_query):
	return griddata( grid, groundTruth, X_query)

def generateStiffnessMap(Xg,Yg):
	
	m1=[20.0,20.0]
	s1=60.0*np.identity(2)
	m2=[30.0,60.0]
	s2=60.0*np.identity(2)
	m3=[60.0,60.0]
	s3=150.0*np.identity(2)
	m4=[70.0,20.0]
	s4=60*np.identity(2);

	grid = np.array([Xg.flatten(), Yg.flatten()]).T
	
	mvn1 = multivariate_normal(m1,s1)
	G1 = mvn1.pdf(grid)
	mvn2 = multivariate_normal(m2,s2)
	G2 = mvn2.pdf(grid)
	mvn3 = multivariate_normal(m3,s3)
	G3 = mvn3.pdf(grid)
	mvn4 = multivariate_normal(m4,s4)
	G4 = mvn4.pdf(grid)

	G=G1+G2+2*G3+G4
	# G=np.max(G,0); #crop below 0
	G[G<0.0]=0.0
	G=G/np.max(G); #normalize
	return grid, G

grid,yGt = generateStiffnessMap(Xg,Yg)

#visualize ground truth
plt.figure(1)
visualize_map(yGt,'Ground Truth')


# data0= grid + .5
# Z0 = evaluateStiffness(grid=grid, groundTruth=yGt, X_query=data0)
# visualize_map(Z0)

X=grid[np.random.randint(0,10000,[100,1]),:]
X=X.reshape(100,-1)
y=evaluateStiffness(grid=grid, groundTruth=yGt, X_query=X)

kernel = C(1.0, (1e-3, 1e3)) * RBF(12, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
probedPoints=[]
stiffnessCollected=[]
for ind in range(X.shape[0]):	
	xind=X[ind,:]
	probedPoints.append(xind.tolist())
	# embed()
	yind=evaluateStiffness(grid=grid, groundTruth=yGt, X_query=xind)
	stiffnessCollected.append(yind.tolist())

	probedPoints_array = np.asarray(probedPoints)
	stiffnessCollected_array = np.asarray(stiffnessCollected)
	gp.fit(probedPoints_array, stiffnessCollected_array)
	y_pred, sigma = gp.predict(grid, return_std=True)
	plt.figure(2)
	visualize_map(y_pred,'Estimated map',probed_points=probedPoints_array)

plt.show()
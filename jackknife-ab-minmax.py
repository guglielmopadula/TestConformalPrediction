import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.subsample import Subsample

from time import time
np.random.seed(0)
num_data=np.array((50,100,500,1000,5000))
sigma=np.array([0,0.001,0.01,0.05,0.1])
results=np.zeros((5,6))

x_test=1+0.5*np.random.rand(1000).reshape(-1,1)
y_test=np.exp(x_test)


start=time()

for i in range(len(num_data)):
    for j in range(len(sigma)):
        x=1+0.5*np.random.rand(num_data[i])
        y=np.exp(x)+sigma[j]*np.random.randn(num_data[i])
        gp = GaussianProcessRegressor()
        mp=MapieRegressor(gp, method='minmax',cv=Subsample(n_resamplings=50))
        mp.fit(x.reshape(-1,1), y)
        y_pred, y_pis = mp.predict(x_test, alpha=0.05)
        results[i,j]=0.5*np.mean(np.abs((y_pred-y_pis[:,0,:].reshape(-1))/y_pred))+0.5*np.mean(np.abs((y_pred-y_pis[:,1,:].reshape(-1))/y_pred))
end=time()
print(end-start)
print(np.mean(results))


import matplotlib.pyplot as plt
plt.plot(num_data,results[:,0],label='sigma=0')
plt.plot(num_data,results[:,1],label='sigma=0.001')
plt.plot(num_data,results[:,2],label='sigma=0.01')
plt.plot(num_data,results[:,3],label='sigma=0.05')
plt.plot(num_data,results[:,4],label='sigma=0.1')
plt.legend()
plt.savefig('jackknife_ab_minmax.png')

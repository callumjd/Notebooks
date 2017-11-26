
# coding: utf-8

# In[16]:


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF


# In[17]:


def get_fps(mol):
    calc=MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    ds = np.asarray(calc.CalcDescriptors(mol))
    
    fps=AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=1024)
    arr=np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fps, arr)
    return np.append(arr,ds)


# In[18]:


#Read the data
data = pd.read_table('smi_sol.dat', sep=' ')
 
#Add some new columns
data['Mol'] = data['smiles'].apply(Chem.MolFromSmiles)
data['Descriptors'] = data['Mol'].apply(get_fps)


# In[19]:


data.head()


# In[20]:


#Convert to Numpy arrays
X = np.array(list(data['Descriptors']))
y = data['solubility'].values
 
#Scale X to unit variance and zero mean
st = StandardScaler()
X= st.fit_transform(X)
 
#Divide into train and test set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=42)


# In[21]:


#from sklearn.model_selection import GridSearchCV

#GPmodel = GridSearchCV(gaussian_process.GaussianProcessRegressor(normalize_y=True), cv=20,
#              param_grid={"alpha": np.logspace(-15, -10, 30),}, scoring='neg_mean_absolute_error', n_jobs=-1)
#GPmodel = GPmodel.fit(X_train, y_train)
#Best_GaussianProcessRegressor = GPmodel.best_estimator_
#print("Best Gaussian Process model")
#print(GPmodel.best_params_)
#print(-1*GPmodel.best_score_)


# In[22]:


# Gaussian process


# In[23]:


kernel=1.0 * RBF(length_scale=1) + WhiteKernel(noise_level=1)


# In[24]:


gp = gaussian_process.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0,normalize_y=True)
gp.fit(X_train, y_train)


# In[25]:


y_pred, sigma = gp.predict(X_test, return_std=True)
rms = (np.mean((y_test - y_pred)**2))**0.5
#s = np.std(y_test -y_pred)
print "GP RMS", rms


# In[26]:


plt.scatter(y_train,gp.predict(X_train), label = 'Train', c='blue')
plt.title('GP Predictor')
plt.xlabel('Measured Solubility')
plt.ylabel('Predicted Solubility')
plt.scatter(y_test,gp.predict(X_test),c='lightgreen', label='Test', alpha = 0.8)
plt.legend(loc=4)
plt.show()


# In[27]:


# Random Forest


# In[ ]:


#RFmodel = GridSearchCV(RandomForestRegressor(), cv=20,
#              param_grid={"n_estimators": np.linspace(50, 150, 25).astype('int')}, scoring='neg_mean_absolute_error', n_jobs=-1)

#RFmodel = RFmodel.fit(X_train, y_train)
#Best_RandomForestRegressor = RFmodel.best_estimator_
#print("Best Random Forest model")
#print(RFmodel.best_params_)
#print(-1*RFmodel.best_score_)


# In[28]:


rf = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=123456)
rf.fit(X_train, y_train)


# In[29]:


y_pred = rf.predict(X_test)
rms = (np.mean((y_test - y_pred)**2))**0.5
print "RF RMS", rms


# In[30]:


plt.scatter(y_train,rf.predict(X_train), label = 'Train', c='blue')
plt.title('RF Predictor')
plt.xlabel('Measured Solubility')
plt.ylabel('Predicted Solubility')
plt.scatter(y_test,rf.predict(X_test),c='lightgreen', label='Test', alpha = 0.8)
plt.legend(loc=4)
plt.show()


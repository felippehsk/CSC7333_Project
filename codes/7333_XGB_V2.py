#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from xgboost import plot_importance


# In[2]:


pip install xgboost


# In[3]:


#Dataset importing and display head
dataset = pd.read_csv('Friday-WorkingHours-Morning.pcap_ISCX.csv')
dataset.head()


# In[4]:


def con_to_bin(x):
    if x=='BENIGN':
        return 0
    if x=='Bot':
        return 1
    
dataset['Label'] = dataset[' Label'].apply(con_to_bin)


# In[5]:


dataset.Label.value_counts()


# In[6]:


dataset.head()


# In[7]:


df = dataset.drop(columns=[' Destination Port', ' Label'], axis=1)
df.head()


# In[8]:


#Assigning X- independent and y- dependent variable
X = df.drop(columns='Label')
y = df.iloc[:,77]
print(X.shape)
print(y.shape)


# In[ ]:





# In[9]:


#Importing Randomsearch cross validation
from sklearn.model_selection import RandomizedSearchCV


# In[10]:


import xgboost as xgb
classifier=xgb.XGBClassifier()


# In[11]:


#Hyperparameter optimization
parameters={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15, 18, 20],
 "min_child_weight" : [ 1, 3, 5, 7, 8 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4, 0.5 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7, 0.9 ]   
}


# In[16]:


#providing XGboost classifier to random search cross validation 
random_search=RandomizedSearchCV(classifier,param_distributions=parameters,n_iter=5,scoring='recall',n_jobs=-1,cv=10,verbose=3)


# In[17]:


#Implementing random search CV by providing input features
random_search.fit(X,y)


# In[18]:


#paramaters estimator and optimized results
random_search.best_estimator_


# In[19]:


#Assigning an object to XGBclassifier
gbm = xgb.XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0.5, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.3, max_delta_step=0, max_depth=20,
              min_child_weight=1, monotone_constraints=None,
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)


# In[20]:


from sklearn.metrics import accuracy_score 


# In[ ]:


#Importing Stratified K fold cross validation and Standard Scaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

#Implementing Stratified K fold CV
accuracy=[]
skf=StratifiedKFold(n_splits=100, random_state=None)
skf.get_n_splits(X,y)

for train_index, test_index in skf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
#Standardizing after splitting dataset
    
    X_norm_train = pd.DataFrame(StandardScaler().fit_transform(X_train), columns = X_train.columns)
    
    gbm.fit(X_norm_train,y_train)
    
    X_norm_test = pd.DataFrame(StandardScaler().fit_transform(X_test), columns = X_test.columns)
    
    predictions = gbm.predict(X_norm_test)
    score_=accuracy_score(predictions,y_test)
    accuracy.append(score_)
    
print(accuracy) 
   


# In[ ]:


#Accuracy evaluation
np.array(accuracy).mean()


# In[ ]:


gbm.score(X_norm_test, y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))


# In[ ]:


print('X_norm_train:', X_norm_train.shape)
print('y_train:', y_train.shape)
print('X_norm:', X_norm_test.shape)
print('y_test:', y_test.shape)


# In[ ]:


predictions.shape


# In[ ]:


from pylab import rcParams
plot_importance(gbm, max_num_features=10)
rcParams['figure.figsize'] = 4,4


# In[ ]:


#Evaluating top features
cols_list = list(X.columns)
feature_importances = pd.DataFrame(gbm.feature_importances_,
                                   index = cols_list,
                                    columns=['F1 score']).sort_values('F1 score', ascending=False)
feature_importances.head(10)


# In[ ]:





# In[ ]:





import numpy as np
import csv
import pandas as pd
import math

from sklearn import linear_model
from sklearn import cross_validation, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance,XGBRegressor
from matplotlib import pyplot as plt

#Train Data
train_data = pd.read_csv("Train.csv")
x = train_data.describe()
df = pd.DataFrame()

df = train_data.iloc[:,5:34]
df = df.drop(columns="p_em_total")
df = df.fillna(df.mean())

#scaling of features is done on per capita basis
for i in range(0,18):
    if i<12:
       df.iloc[:,i] = df.iloc[:,i]/train_data.iloc[:,4]
    elif 12<=i<14:
       continue
    elif 14<=i<18:
       df.iloc[:,i] = ((df.iloc[:,i])*1000)/((train_data.iloc[:,19])*train_data.iloc[:,4])




#Storing features and targets in array    
X = np.array(df)
y = np.array(train_data['per_capita_exp_total_py'])

#Splitting data into train and test 
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X, y, test_size = 0,
                                     random_state = 42)

#Applying XGBOOST regression and tuning parameters
import xgboost

xgb = xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.08, gamma=0, subsample=1,
                           colsample_bytree=1, max_depth=7,seed=100,min_child_weight=100)





#fitting model and predicting
xgb.fit(X_train, y_train)


"""
prediction = xgb.predict(X_test)

#Calculating R^2
print(xgb.score(X_test, y_test))

#Root mean Squared error
mse = mean_squared_error(prediction, y_test)
rmse = np.sqrt(mse)
print(rmse)

#plot feature importance
plot_importance(xgb)
plt.show()
"""

# Predicting missing values of "per_capita_exp_total_py" (Test Data)
test_data = pd.read_csv("Test.csv")
key0 = test_data.iloc[:,0]
key1 = test_data.iloc[:,1]
df2 = pd.DataFrame()
df2 = test_data.iloc[:,5:34]
df2 = df2.drop(columns="p_em_total")
df2 = df2.fillna(df.mean())

#scaling of features is done on per capita basis
for i in range(0,18):
    if i<12:
       df2.iloc[:,i] = df2.iloc[:,i]/test_data.iloc[:,4]
    elif 12<=i<14:
       continue
    elif 14<=i<18:
       df2.iloc[:,i] = ((df2.iloc[:,i])*1000)/((test_data.iloc[:,19])*test_data.iloc[:,4])

"""
#Changing Variables to Reduce Expenditure per Capita py
df2.iloc[:,12] = df2.iloc[:,12]*0.7
df2.iloc[:,16] = df2.iloc[:,14]*0.7
df2.iloc[:,16] = df2.iloc[:,15]*0.7
df2.iloc[:,16] = df2.iloc[:,16]*0.7
df2.iloc[:,16] = df2.iloc[:,17]*0.7
"""


data_test = np.array(df2)
#predicting on Test Data
pred = xgb.predict(data_test)
key2 = np.array([round(x) for x in pred])
key0 = np.array(key0)
key1 = np.array(key1)


key2 = np.array([key0]+[key1]+[key2])
key2 = key2.transpose()
#Exporting predicted per_capita_exp_total_py to Solution.csv
column =["aco_num","aco_name","per_capita_exp_total_py"]
df3 = pd.DataFrame(key2, columns= column)
df3.to_csv("Solution.csv", index = False)




	



import pandas as pd
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import numpy as np
import matplotlib.pyplot as plt



#read the data
#data = pd.read_csv("Data/TrainingData/training.csv", header=0, sep=';')
data = pd.read_csv("Data/TestingData/testing9.csv", header=0, sep=';')


del data['DateTime']
del data['Speed']
del data['LiftWorkingPosition']
# data["LiftWorkingPosition"] = data["LiftWorkingPosition"].astype(int)



# df['Date_Time'] = pd.to_datetime(df.Date_Time , format = '%d/%m/%Y %H.%M.%S')
# data = df.drop(['Date_Time'], axis=1)
# data.index = df.Date_Time

# #missing value treatment
cols = data.columns
# for j in cols:
#     for i in range(0,len(data)):
#        if data[j][i] == -200:
#            data[j][i] = data[j][i-1]


# #checking stationarity
from statsmodels.tsa.vector_ar.vecm import coint_johansen
# #since the test works for only 12 variables, I have randomly dropped
# #in the next iteration, I would drop another and check the eigenvalues
# johan_test_temp = data.drop([ 'CO(GT)'], axis=1)
# coint_johansen(johan_test_temp,-1,1).eig

data = data.to_numpy()

type(data)
#creating the train and validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR


train.shape

model = VAR (endog=train)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))

#converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,5):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

#check rmse

pred.head


data.iloc[0:376].plot(y = "Current")
pred.plot(y = "Current")
plt.show()

# tempData.plot()
# plt.show()



for i in cols:
    print('rmse value for' + i + 'is : ', sqrt(mean_squared_error(pred[pred.columns.get_loc(i)], valid[pred.columns.get_loc(i)])))


#make final predictions
model = VAR(endog=data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)
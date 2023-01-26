import statsmodels.formula.api as smf
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

data_path = 'path'

train1_90 = pd.read_csv(data_path+'/train1_90.csv')
# train1_90 = csv.reader(open('train1_90.csv', 'r'), delimiter=",", quotechar='|')
train1_180 = pd.read_csv(data_path+'/train1_180.csv')
# train1_180 = csv.reader(open('train1_180.csv', 'r'), delimiter=",", quotechar='|')
train1_360 = pd.read_csv(data_path+'/train1_360.csv')
# train1_360 = csv.reader(open('train1_360.csv', 'r'), delimiter=",", quotechar='|')

train2_90 = pd.read_csv(data_path+'/train2_90.csv')
train2_180 = pd.read_csv(data_path+'/train2_180.csv')
train2_360 = pd.read_csv(data_path+'/train2_360.csv')

test_90 = pd.read_csv(data_path+'/test_90.csv')
test_180 = pd.read_csv(data_path+'/test_180.csv')
test_360 = pd.read_csv(data_path+'/test_360.csv')

def similarity(a,b):
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    nr = np.dot(a - a_mean,b - b_mean)
    dr = len(a) * np.std(a) * np.std(b)
    similarity = nr/dr
    return similarity


def computeDelta(wt, X, Xi):
    similarity_vector = Xi.iloc[:,:-1].apply(lambda x : similarity(x,X[:-1]),axis = 1)
    similarity_exp = np.exp(wt * similarity_vector)
    price_change = np.dot(Xi.iloc[:,-1],similarity_exp)/np.sum(similarity_exp)
    return price_change


weight = 2
trainDeltaP90 = np.empty(0)
trainDeltaP180 = np.empty(0)
trainDeltaP360 = np.empty(0)
for i in range(0,len(train1_90.index)) :
  trainDeltaP90 = np.append(trainDeltaP90, computeDelta(weight,train2_90.iloc[i],train1_90))
for i in range(0,len(train1_180.index)) :
  trainDeltaP180 = np.append(trainDeltaP180, computeDelta(weight,train2_180.iloc[i],train1_180))
for i in range(0,len(train1_360.index)) :
  trainDeltaP360 = np.append(trainDeltaP360, computeDelta(weight,train2_360.iloc[i],train1_360))


trainDeltaP = np.asarray(train2_360[['Yi']])
trainDeltaP = np.reshape(trainDeltaP, -1)

d = {'deltaP': trainDeltaP,
     'deltaP90': trainDeltaP90,
     'deltaP180': trainDeltaP180,
     'deltaP360': trainDeltaP360 }
trainData = pd.DataFrame(d)

model = smf.ols(formula = 'deltaP ~ deltaP90 + deltaP180 + deltaP360',data = trainData).fit()

print(model.params)


testDeltaP90 = np.empty(0)
testDeltaP180 = np.empty(0)
testDeltaP360 = np.empty(0)
for i in range(0,len(train1_90.index)) :
  testDeltaP90 = np.append(testDeltaP90, computeDelta(weight,test_90.iloc[i],train1_90))
for i in range(0,len(train1_180.index)) :
  testDeltaP180 = np.append(testDeltaP180, computeDelta(weight,test_180.iloc[i],train1_180))
for i in range(0,len(train1_360.index)) :
  testDeltaP360 = np.append(testDeltaP360, computeDelta(weight,test_360.iloc[i],train1_360))


testDeltaP = np.asarray(test_360[['Yi']])
testDeltaP = np.reshape(testDeltaP, -1)


d = {'deltaP': testDeltaP,
     'deltaP90': testDeltaP90,
     'deltaP180': testDeltaP180,
     'deltaP360': testDeltaP360}
testData = pd.DataFrame(d)


result = model.predict(testData)
compare = { 'Actual': testDeltaP,
            'Predicted': result }
compareDF = pd.DataFrame(compare)

df = pd.read_csv(data_path+'/dataset.csv')
df.plot(kind = 'line', x = 'time', y = 'price')
plt.show()

MSE = 0.0
MSE = sm.mean_squared_error(y_true=testDeltaP,y_pred=result)
print("The MSE is %f" % (MSE))

df = pd.read_csv(data_path+'/dataset.csv')
df.plot(kind = 'line', x = 'time', y = 'price')
plt.show()


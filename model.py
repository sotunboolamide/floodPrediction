## Reading libraries
import pandas as pd
import numpy as np
import warnings
from sklearn import metrics
from sklearn.metrics import mean_squared_error,log_loss
#import requests
from io import StringIO 
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.tree import DecisionTreeClassifier


train  = pd.read_csv("dataset/pluvialflood.csv")

def convert(x):
  if x =='Low':
    return 1
  elif x =='Moderate':
    return 2
  elif x == 'High':
    return 3
  elif x =='Very_High':
    return 4
  else:
    return 0

train['SUSCEP'] = train['SUSCEP'].apply(convert)

X = train.drop(['SUSCEP'],axis =1)
y = train['SUSCEP']

from sklearn.model_selection import train_test_split
#from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

 
tree = DecisionTreeClassifier(max_depth=3, random_state=17)
tree.fit(X_train,y_train)
from sklearn.linear_model import LogisticRegression 
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)


pickle.dump(clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

print(model.predict([[4,7,3,2,1,4,1,5,5]]))
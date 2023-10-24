import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib

consumption=pd.read_csv('measurements.csv')
consumption['distance']=consumption['distance'].str.replace(',','.').astype(float)
consumption['consume']=consumption['consume'].str.replace(',','.').astype(float)
consumption['temp_inside']=consumption['temp_inside'].str.replace(',','.').astype(float)
data=consumption.drop(['specials','temp_outside','refill liters','refill gas'],axis=1)
data['temp_inside']=data['temp_inside'].fillna(data['temp_inside'].mean())
from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()
data['gas_type']=lr.fit_transform(data['gas_type'])
# data['distance']=data['distance'].astype(int)
# data['consume']=data['consume'].astype(int)
# data['temp_inside']=data['temp_inside'].astype(int)
x=data.drop('consume',axis=1)
y=data.iloc[:,1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=0)
# x_train.to_csv('x_train.csv',index=False)

from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
x_trainmn=mms.fit_transform(x_train)
x_testmn=mms.transform(x_test)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_trainstd=ss.fit_transform(x_train)
x_teststd=ss.transform(x_test)
from sklearn.decomposition import PCA
pc=PCA(n_components=5)
x_trainpca=pc.fit_transform(x_train)
x_testpca=pc.transform(x_test)

from sklearn.tree import DecisionTreeRegressor
# dt1=DecisionTreeRegressor(max_depth=3,random_state=56)
# dt1.fit(x_train,y_train)
# joblib.dump(dt1, 'decision_tree_model_normal.pkl')
# pred1=dt1.predict(x_test)

# dt2=DecisionTreeRegressor(max_depth=3,random_state=56)
# dt2.fit(x_trainmn,y_train)
# joblib.dump(dt2, 'decision_tree_model_minmax.pkl')
# pred2=dt2.predict(x_testmn)

# dt3=DecisionTreeRegressor(max_depth=3,random_state=56)
# dt3.fit(x_trainstd,y_train)
# joblib.dump(dt3, 'decision_tree_model_standard.pkl')
# pred3=dt3.predict(x_teststd)

dt4=DecisionTreeRegressor(max_depth=3,random_state=56)
dt4.fit(x_trainpca,y_train)
joblib.dump(dt4, 'decision_tree_model_pca.pkl')
pred4=dt4.predict(x_testpca)

# from sklearn.ensemble import RandomForestClassifier
# model1 = RandomForestClassifier(n_estimators=5, random_state=50,criterion='entropy',max_depth=4)
# model1.fit(x_train, y_train)
# joblib.dump(model1, 'random_forest_model_normal.pkl')
# pred5=model1.predict(x_test)

# model4= RandomForestClassifier(n_estimators=5, random_state=50,criterion='entropy',max_depth=4)
# model4.fit(x_trainpca, y_train)
# joblib.dump(model4, 'random_forest_model_pca.pkl')
# pred6=model4.predict(x_testpca)

# from sklearn.linear_model import LinearRegression
# lnr=LinearRegression() 
# lnr.fit(x_train,y_train)
# joblib.dump(lnr, 'lỉnear_regression_model_normal.pkl')
# pd=lnr.predict(x_test)

# lnr1=LinearRegression()
# lnr1.fit(x_trainmn,y_train)
# joblib.dump(lnr1, 'lỉnear_regression_model_minmax.pkl')
# pd1=lnr.predict(x_testmn)

# lnr2=LinearRegression()
# lnr2.fit(x_trainstd,y_train)
# joblib.dump(lnr2, 'lỉnear_regression_model_standard.pkl')
# pd2=lnr2.predict(x_teststd)

# lnr3=LinearRegression()
# lnr3.fit(x_trainpca,y_train)
# joblib.dump(lnr3, 'lỉnear_regression_model_pca.pkl')
# pd3=lnr3.predict(x_testpca)


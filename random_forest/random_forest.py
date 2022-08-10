# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:38:39 2022

@author: EBRU
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df= pd.read_csv("C:\\Users\\EBRU\\Desktop\\machine learning\\random_forest\\random_forest_dataset.csv", sep=";", header=None)
x= df.iloc[:,0].values.reshape(-1,1) # tüm satırlar ve sadece 0. sütun'u al
y= df.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=100, random_state= 42) 
# n_estimators= sub_data'da kaç tane tree kullanmak istediğimizi yazıyoruz.
# random_state ise datadan n sayıda sample seçimi yapılırken random şekilde  yapıyorduk.
# ama kodu iki kere çalıştırdığımızı düşününce tekrar bir n sayıda seçim yapılacak ve bir önceki seçim sayısından farklı olabilir.
# random_state değeri belirlersek id gibi bir değer tutmuş olacağız ve her zaman o sayıda seçim yapılacak.
# bu değer genelde 42 olarak seçildiği için 42 yazdık.

rf.fit(x,y)
print("7,8 seviyesinde fiyatın ne kadar olduğunun tahmini: ", rf.predict([[7.8]]))

x_= np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head= rf.predict(x_)

#visualization
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()

# decision_tree'den farkı 1 adet ağaca bakmak yerine n_estimator ile belirlediğimiz sayıda ağaca bakılabiliyor.
# 100 tane decision tree'ye bakmak 1 tanesine bakmaktan daha iyidir.
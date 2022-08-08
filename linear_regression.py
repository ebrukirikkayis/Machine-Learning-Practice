# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt

#%%
df= pd.read_csv("C:\\Users\\EBRU\\Desktop\\machine learning\\linear_regression_dataset.csv", sep=";")

plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%% linear regression

from sklearn.linear_model import LinearRegression

linear_reg= LinearRegression()

x= df.deneyim.values.reshape(-1,1)
y= df.maas.values.reshape(-1,1)
# reshape yapıyoruz çünkü yapmadan önce console'da x.shape olarak çalıştırdığımızda (14,) olarak kaydediyor,
# ama (14,)' boşluğun 1 olduğunu sklearn anlamaz bu yüzden -1,1 aralığında reshape ediyoruz.

linear_reg.fit(x,y)

#%%
import numpy as np
b0= linear_reg.predict([[0]]) #line'ın y eksenini kestiği nokta
print("b0",b0)

b0_=linear_reg.intercept_   # b0'ı bulmak için özel bir metod
print("bo_:",b0_)  

b1= linear_reg.coef_
print("b1",b1)

maas_yeni=1302 + 1138*11 
print(maas_yeni)

print(linear_reg.predict([[11]]))   # 11 yıllık deneyim için maaş tahmini

array= np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
plt.scatter(x,y)
plt.show()

y_head= linear_reg.predict(array)
plt.plot(array, y_head, color="red")
plt.scatter(x,y)
plt.show()

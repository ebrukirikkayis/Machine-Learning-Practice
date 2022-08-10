# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 22:54:45 2022

@author: EBRU
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("C:\\Users\\EBRU\\Desktop\\machine learning\\r_square\\r_square_linear_regression_dataset.csv", sep=";")

#%%
from sklearn.linear_model import LinearRegression

linear_reg= LinearRegression()
x= df.deneyim.values.reshape(-1,1) #1.sütun
y= df.maas.values.reshape(-1,1) #2.sütun 

plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
linear_reg.fit(x,y)
y_head= linear_reg.predict(x)  #maas
plt.plot(x,y_head,color="red")
plt.show()

#%%
from sklearn.metrics import r2_score
print("r_square score: ", r2_score(y,y_head))
# sonuç 0.97 çıkıyor bu demek oluyor ki model iyi bir tahminde bulunmuştur.


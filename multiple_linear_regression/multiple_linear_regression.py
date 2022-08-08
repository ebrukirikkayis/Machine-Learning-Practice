# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 17:12:48 2022

@author: EBRU
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df= pd.read_csv("C:\\Users\\EBRU\\Desktop\\machine learning\\multiple_linear_regression\\multiple_linear_regression_dataset.csv", sep=";")
x= df.iloc[:,[0,2]].values # tüm satırlar ile 0 ve 2. column'ı al
y= df.maas.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0:", multiple_linear_regression.intercept_)  #b0'ın değerini yazdıralım (bias-constant)
print("b1, b2:", multiple_linear_regression.coef_) 

# Predict
print("predict: ", multiple_linear_regression.predict(np.array([[10,35],[5,35]])))
# verilen yaşlar aynı olmasına rağmen 10 yıllık deneyimi olan tahminde 5 yıllık deneyime göre yaklaşık 3 kat daha fazla maaş almaktadır.

# multiple linear regression'da birden fazla independent variable bulunup dependent variable'ı etkiliyor.
# linear regressionda ise sadece bir tane independent variable bulunablir.
# metodumuz MSE(min square error)
# implementation için   LinearRegression() kullandık.
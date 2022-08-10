# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 22:34:17 2022

@author: EBRU
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("C:\\Users\\EBRU\\Desktop\\machine learning\\r_square\\r_square_random_forest_dataset.csv", sep=";")

x= df.iloc[:,0].values.reshape(-1,1) #1.sütun
y= df.iloc[:,1].values.reshape(-1,1) #2.sütun

#%%
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x,y)

y_head= rf.predict(x)
# x için değerler predict edildi şimdi bu değerlerin doğruluğunu test etmek için 
# r square metodu kullanılacak

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# burada anlamadığım bir şey var predict metodu neden x'i parametre olarak aldı
# y üzerinden tahmin yapılıyorsa sadece y'yi alması gerekmez miyd ?


#%%
from sklearn.metrics import r2_score
# python'da metrics'ler içerisinde r square hesaplayan metod bulunmaktadır.
print("r_score:", r2_score(y,y_head))
# gerçek y değerleri ile predict edilen y_head değerlerini parametre olarak veriyoruz.
# bunun sonucu burada 0.97 çıktı (1'e çok yakın ve bu demek oluyorki yapılan tahmin başarılı)

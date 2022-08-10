# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 23:00:43 2022

@author: EBRU
"""

import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv("C:\\Users\\EBRU\\Desktop\\machine learning\\polynomial_linear_regression\\polynomial_linear_regression_dataset.csv", sep=";")

x= df.araba_fiyat.values.reshape(-1,1)
y= df.araba_max_hiz.values.reshape(-1,1)

# values series type'ını array'e çevirirken kullanılır.
# sklearn'da bu şekilde tanımlama yapabilmek için reshape özelliği kullanılır.(15,)'ü (15,1)'e çevirir ve sklearn bunu anlayabilir.

plt.scatter(x,y)
plt.xlabel("araba_fiyat")
plt.ylabel("araba_max_hiz")
plt.show()

# linear regression = y=b0+b1*x
# multiple linear regression = y=b0+b1*x1 + b2*x2

#%% linear regression
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x,y)
#%%
y_head= lr.predict(x)

plt.scatter(x,y)
plt.xlabel("araba_fiyat")
plt.ylabel("araba_max_hiz")
plt.plot(x,y_head, color="red")
plt.show()
#%%
print(lr.predict([[10000]]))
# 10000 tl olan bir arabanın max_hız'ı ne olur sorusuna karşılık olarak bu şekilde bir prediction yapıldı.
# bunun sonucunda çıkan hız değeri çok yüksek ve bir arabanın o hıza ulaşmasın çok zor
# bu yüzden bu data linear regression içi uygun değildir deriz. 
# bu data polinom şeklinde olmalı. Heaplamalar da öyle yapılmalı.
#%%
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression= PolynomialFeatures(degree=2) 
# degree n'in kaça kadar olacağını belirler. n burada b0+b1*x1+b2*x^2+...+bn*x^n 'dir.

x_polynomial= polynomial_regression.fit_transform(x)
# burada sadece fit yaptığımızda polynomial işlemler yapılır fakat bize bir işlem geri dönmez PolynomialFeatures metodu geri döner.
# bu yüzden fit_transform metodunu kullanıp içerisine x parametresini veriyoruz.
# transform burada uygulanan işlemin x_polynomial'a çevirilmesini sağlar. 
# x parametresi verideki araba_fiyat feature'ına sahipti 
# bu şekilde PolynomialFeatures metodu araba_fiyat feature'ının 2.dereceden hesaplamasını gerçekleştirir.

#print(x_polynomial)
# şimdi ise MSE bulmaya çalışalım.

linear_regression2= LinearRegression()
# formülümüz b0+b1*x1+b2*x^2+...+bn*x^n olup yukarıda x_polynomial'da data'nın x'leri hesaplandığından dolayı
# şimdi elde ettiğimiz yeni x'ler ile y'leri fit edip model oluşturalım.
linear_regression2.fit(x_polynomial,y) 

y_head2= linear_regression2.predict(x_polynomial)

plt.plot(x,y_head, color="red", label="linear" )
plt.plot(x,y_head2,color="green", label="poly")
plt.scatter(x,y)
plt.legend()
plt.show()
# poly modeli elde edildi ve burada MSE azaltılmış oldu.
#%%
# degree'yi 4 yaparsak model daha karmaşıklaşacak ve daha iyi tahmin yapacak.
polynomial_regression2= PolynomialFeatures(degree=4) 
x_polynomial= polynomial_regression2.fit_transform(x)
linear_regression3= LinearRegression()
linear_regression3.fit(x_polynomial,y) 
y_head3= linear_regression3.predict(x_polynomial)
plt.plot(x,y_head, color="red", label="linear" )
plt.plot(x,y_head2,color="green", label="poly")
plt.plot(x,y_head3,color="black", label="poly")
plt.scatter(x,y)
plt.legend()
plt.show()

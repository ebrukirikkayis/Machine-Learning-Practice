# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 09:22:02 2022

@author: EBRU
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df= pd.read_csv("C:\\Users\\EBRU\\Desktop\\machine learning\\decision_tree\\decision_tree_dataset.csv", sep=";")

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)
#%% decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg= DecisionTreeRegressor() # class'ı verdik
tree_reg.fit(x,y) # modeli oluşturduk

print(tree_reg.predict([[6]]))
# 6 değeri veride 40 değerine sahip burada da tahmin olarak 40 sonucunu aldık
# tahminimiz doğru demektir.
print(tree_reg.predict([[5.5]]))
# 5 değeri veride 50'ye denk geliyor. Burada da tahmin 50 olarak yapıldı. 
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head= tree_reg.predict(x_)

# x_ değişkenini yaratmamızın nedeni decision tree'nin belli bir aralığa kadar aynı değeri veriyor olması gerektiğidir.
# yeni bir aralığa geçildiğinde yeni bir sonuca geçileceği için aralık olarak verdik.
#%% visualization

plt.scatter(x,y,color="red")
plt.plot(x_,y_head, color="green")
plt.xlabel("tribün level")
plt.ylabel("ücret")
plt.show()

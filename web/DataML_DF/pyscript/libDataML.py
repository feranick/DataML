# -*- coding: utf-8 -*-
'''
**************************************************
* libDataML - Library for DataML/DataML_DF
* v2024.11.03.1
* Uses: Keras, TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
**************************************************
'''
import numpy as np

#************************************
# MultiClassReductor
#************************************
class MultiClassReductor():
    def __self__(self):
        self.name = name
    
    def fit(self,tc):
        self.totalClass = tc.tolist()
    
    def transform(self,y):
        Cl = np.zeros(y.shape[0])
        for j in range(len(y)):
            Cl[j] = self.totalClass.index(np.array(y[j]).tolist())
        return Cl
    
    def inverse_transform(self,a):
        return [self.totalClass[int(a[0])]]
        
    def inverse_transform_bulk(self,a):
        inv=[]
        for i in range(len(a)):
            inv.append(self.totalClass[int(a[i])])
        return inv

    def classes_(self):
        return self.totalClass

# -*- coding: utf-8 -*-
'''
**************************************************
* libDataML - Library for DataML/DataML_DF
* v2025.09.15.1
* Uses: Keras, TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
**************************************************
'''
import numpy as np

#************************************
# Normalizer
#************************************
class Normalizer(object):
    def __init__(self, M, dP):
        self.M = M
        self.normalizeLabel = dP.normalizeLabel
        self.useCustomRound = False
        self.minGeneralLabel = 0
        self.maxGeneralLabel = 1
        self.YnormTo = 1
        self.stepNormLabel = 0.01
        self.saveNormalized = True
        self.norm_file = "norm_file.pkl"
        
        self.data = np.arange(0,1,self.stepNormLabel)
        self.min = np.zeros([self.M.shape[1]])
        self.max = np.zeros([self.M.shape[1]])
    
        if self.normalizeLabel:
            self.min[0] = np.nanmin(self.M[1:,0])
            self.max[0] = np.nanmax(self.M[1:,0])
        
        for i in range(1,M.shape[1]):
            self.min[i] = np.nanmin(self.M[1:,i])
            self.max[i] = np.nanmax(self.M[1:,i])
    
    def transform(self,y):
        Mn = np.copy(y)
        if self.normalizeLabel:
            if self.max[0] - self.min[0] == 0:
                Mn[1:,0] = (self.max[0] + self.min[0])/2
            else:
                Mn[1:,0] = np.multiply(y[1:,0] - self.min[0],
                    self.YnormTo/(self.max[0] - self.min[0]))
            if self.useCustomRound:
                customData = CustomRound(self.data)
                for i in range(1,y.shape[0]):
                    Mn[i,0] = customData(Mn[i,0])
        if self.saveNormalized:
            for i in range(1,y.shape[1]):
                if self.max[i] - self.min[i] == 0:
                    Mn[1:,i] = (self.max[i] + self.min[i])/2
                else:
                    Mn[1:,i] = np.multiply(y[1:,i] - self.min[i],
                        self.YnormTo/(self.max[i] - self.min[i]))
        return Mn
        
    def transform_valid(self,V):
        Vn = np.copy(V)
        for i in range(0,V.shape[0]):
            if self.max[i+1] - self.min[i+1] == 0:
                Vn[i,1] = (self.max[i+1] + self.min[i+1])/2
            else:
                Vn[i,1] = np.multiply(V[i,1] - self.min[i+1],
                    self.YnormTo/(self.max[i+1] - self.min[i+1]))
        return Vn
    
    def transform_valid_data(self,V):
        Vn = np.copy(V)
        if self.saveNormalized:
            for i in range(0,V.shape[1]):
                if self.max[i+1] - self.min[i+1] == 0:
                    Vn[0][i] = (self.max[i+1] - self.min[i+1])/2
                else:
                    Vn[0][i] = np.multiply(V[0][i] - self.min[i+1],
                    self.YnormTo/(self.max[i+1] - self.min[i+1]))
        return Vn
    
    # Single sample, format [[0,1,2,3,4...]]
    def transform_inverse_single(self,v):
        vn = self.min[0] + v*(self.max[0] - self.min[0])/self.YnormTo
        return vn

    # Multiple samples, format as vertical stacks of singles
    def transform_inverse(self,V):
        Vnt = []
        for i in range (V.shape[0]):
            vn = self.min[0] + V[i]*(self.max[0] - self.min[0])/self.YnormTo
            Vnt.append(vn)
        return np.array(Vnt)

    # format simular to original DataML file
    def transform_inverse_features(self,V):
        Vn = []
        for i in range (V.shape[1]):
            vn = (self.min[i] + + V[1:,i]*(self.max[i] - self.min[i])/self.YnormTo)
            Vn.append(vn)
        return np.vstack([V[0,:],np.vstack(Vn).T])

    def save(self):
        with open(self.norm_file, 'wb') as f:
            pickle.dump(self, f)

#************************************
# MultiClassReductor
#************************************
class MultiClassReductor():
    def __init__(self,dP):
        self.model_le = dP.model_le

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
        
    def save(self):
        with open(self.model_le, 'wb') as f:
            pickle.dump(self, f)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Fit3DPlane
* version: 20210606a
* By: Nicola Ferralis <feranick@hotmail.com>
* Licence: GPL 2 or newer
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, h5py
from sklearn import linear_model

#************************************
# Parameters definition
#************************************
class dP:
    skipHeadRows = 0
#************************************
# Main
#************************************
def main():
    df = readParamFile(sys.argv[1])
    rootFile = os.path.splitext(sys.argv[1])[0]

    x = df["P2"].values
    y = df["P3"].values
    z = df["P1"].values
    
    linear3Dfit(x,y,z)
    quad3Dfit(x,y,z)
    
#************************************
# Linear 3D fit
#************************************
def linear3Dfit(X,Y,Z):
    x1, y1, z1 = X.flatten(), Y.flatten(), Z.flatten()

    X_data = np.array([x1, y1]).reshape((-1, 2))
    Y_data = z1

    reg = linear_model.LinearRegression().fit(X_data, Y_data)
    
    print(" Z = a1 * X + a2 * Y + c")
    print(" a1 =",reg.coef_[0])
    print(" a2 =", reg.coef_[1])
    print(" c: ", reg.intercept_)
    
    print(" R2=",reg.score(X_data, Y_data))

#************************************
# 2nd order 3D fit
#************************************
def quad3Dfit(X,Y,Z):
    x1, y1, z1 = X.flatten(), Y.flatten(), Z.flatten()
    x1y1, x1x1, y1y1 = x1*y1, x1*x1, y1*y1

    X_data = np.array([x1, y1, x1y1, x1x1, y1y1]).T  # X_data shape: n, 5
    Y_data = z1

    reg = linear_model.LinearRegression().fit(X_data, Y_data)
    print("\n Z = a1*X + a2*Y + a3*X*Y + a4*X*X + a5*Y*Y + c")
    print(" a1 =",reg.coef_[0])
    print(" a2 =", reg.coef_[1])
    print(" a3 =",reg.coef_[2])
    print(" a4 =", reg.coef_[3])
    print(" a5 =",reg.coef_[4])
    print(" c = ", reg.intercept_)
    print(" R2=",reg.score(X_data, Y_data))

#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile):
    try:
        with open(paramFile, 'r') as f:
            dfP = pd.read_csv(f, delimiter = ",", skiprows=dP.skipHeadRows)
    except:
        print("\033[1m Param file:",paramFile," not found/broken \n\033[0m")
        return
    return dfP

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

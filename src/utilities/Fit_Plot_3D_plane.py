#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* FitPlot3DPlane
* version: 20210614a
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
    
    xlab = "P1"
    ylab = "P2"
    zlab = "P3"
    tlab = "C9"
    
    showPlots = True
    showFitDataPlots = False
    
#************************************
# Main
#************************************
def main():
    df = readParamFile(sys.argv[1])
    rootFile = os.path.splitext(sys.argv[1])[0]

    dfX = df[dP.xlab]
    dfY = df[dP.ylab]
    dfZ = df[dP.zlab]
    dfT = df[dP.tlab]
    
    m1, c1= linear3Dfit(dfX.values,dfY.values,dfZ.values)
    m2, c2, text = quad3Dfit(dfX.values,dfY.values,dfZ.values)
    
    if dP.showPlots:
        plot(dfX, dfY, dfZ, dfT, m2, c2, text)

#************************************
# Linear 3D fit
#************************************
def linear3Dfit(X,Y,Z):
    x1, y1, z1 = X.flatten(), Y.flatten(), Z.flatten()

    X_data = np.array([x1, y1]).reshape((-1, 2))
    Y_data = z1

    reg = linear_model.LinearRegression().fit(X_data, Y_data)
    r2 = reg.score(X_data, Y_data)
    
    print(" Z = a1 * X + a2 * Y + c")
    print(" a1 =",reg.coef_[0])
    print(" a2 =", reg.coef_[1])
    print(" c =", reg.intercept_)
    print(" R2 =",r2)
    return reg.coef_, reg.intercept_

#************************************
# 2nd order 3D fit
#************************************
def quad3Dfit(X,Y,Z):
    x1, y1, z1 = X.flatten(), Y.flatten(), Z.flatten()
    x1y1, x1x1, y1y1 = x1*y1, x1*x1, y1*y1

    X_data = np.array([x1, y1, x1y1, x1x1, y1y1]).T  # X_data shape: n, 5
    Y_data = z1

    reg = linear_model.LinearRegression().fit(X_data, Y_data)
    r2 = reg.score(X_data, Y_data)
    
    text =" Z = a1*X + a2*Y + a3*X*Y + a4*X*X + a5*Y*Y + c" +\
        "\n a1 = "+str(reg.coef_[0])+"\n a2 = "+str(reg.coef_[1])+\
        "\n a3 = "+str(reg.coef_[2])+"\n a4 = "+str(reg.coef_[3])+\
        "\n a5 = "+str(reg.coef_[4])+"\n c = "+str(reg.intercept_)+\
        "\n R2 = "+str(r2)+"\n"
    
    print("\n",text)
    return reg.coef_, reg.intercept_, text
    
#************************************
# Plot
#************************************
def plot(dfx, dfy, dfz, dft, m, c, text):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    
    fig = plt.figure(figsize=(16, 8))
    
    if dP.showFitDataPlots:
        fig.text(0,0, text)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    s = [s*3 for s in dft.values]
    
    scatter = ax.scatter(dfx.values, dfy.values, dfz.values, s = s)
    
    ax.set_xlim([dfx.min(), dfx.max()])
    ax.set_ylim([dfy.min(), dfy.max()])
    ax.set_zlim([dfz.min(), dfz.max()])
    ax.set_xlabel(dP.xlab)
    ax.set_ylabel(dP.ylab)
    ax.set_zlabel(dP.zlab)
    
    #************************************
    # Plot Surface
    #************************************
    x = np.outer(np.linspace(dfx.min(), dfx.max(), 30), np.ones(30))
    y = np.outer(np.linspace(dfy.min(), dfy.max(), 30), np.ones(30)).T
    z = m[0]*x + m[1]*y + m[2]*x*y + m[3]*x*x + m[4]*y*y + c
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    surf = ax.plot_surface(x,y,z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    ax.set_xlabel(dP.xlab)
    ax.set_ylabel(dP.ylab)
    ax.set_zlabel(dP.zlab)
    
    fig.colorbar(surf, shrink=10.5, aspect=5)
    plt.show()

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

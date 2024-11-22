#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* AddDenoiseAutoEncoder
* Data Augmentation via Denoising Autoencoder
* version: v2024.11.22.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py, pickle
from libDataML import *

#************************************
# Parameters definition
#************************************
class dP:
    saveAsTxt = True
    batch_size = 8
    epochs = 200
    validation_split = 0.1
    l_rate = 0.1
    l_rdecay = 0.01
    numAdditions = 1
    numAddedNoisyDataBlocks = 10
    percNoiseDistrMax = 0.1
    excludeZeroFeatures = True
    numLabels = 1
    normalize = True
    normalizeLabel = True
    norm_file = "norm_file.pkl"
    
#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 AddDenoiseAutoEncoder.py <learnData>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, A, M = readLearnFile(sys.argv[1], True)

    newFile = os.path.splitext(sys.argv[1])[0] + '_numDataTrainDae' + \
        str(dP.numAddedNoisyDataBlocks * A.shape[0]) + '_numAdded' + str(dP.numAdditions * A.shape[0])
            
    newA = createNoysyData(dP, A)
    dae = trainAutoencoder(dP, newA, sys.argv[1])
    newTrain = generateData(dP, dae, En, A, M)
    saveLearnFile(dP, newTrain, newFile, "")

#******************************************************
# Create new Training data by adding a percentage of the max
# for that feature
#******************************************************
def createNoysyData(dP, A):
    import random
    
    newA = np.zeros((0, A.shape[1]))
    A_min = A.min(axis=0)
    A_max = A.max(axis=0)
    A_mean = np.mean(A, axis=0)
    A_std = A.std(axis=0)
        
    for i in range(A.shape[0]):
        for h in range(int(dP.numAddedNoisyDataBlocks)):
            A_tmp = []
            for j in range(A.shape[1]):
                if A[i][j] == 0 and dP.excludeZeroFeatures:
                    tmp = A[i][j]
                else:
                    tmp =  A[i][j]+A_max[j]*(np.random.uniform(-dP.percNoiseDistrMax, dP.percNoiseDistrMax, 1))
                    if tmp<0:
                        tmp=-tmp
                A_tmp = np.hstack([A_tmp, tmp])
            newA = np.vstack([newA, A_tmp])
    '''
    with open(dP.norm_file, "rb") as f:
        norm = pickle.load(f)
    print(norm.transform_inverse(newA))
    saveLearnFile(dP, norm.transform_inverse(newA), "test_noisy_data_for Dae", "")
    '''
    return newA

#************************************
# Train Autoencoder
#************************************
def trainAutoencoder(dP, A, file):
    import keras
    
    input = keras.Input(shape=(A.shape[1],),sparse=True)
    
    ############
    # Encoder
    ############
    encoded = keras.layers.Dense(A.shape[1]-1, activation='relu')(input)
    for i in range(A.shape[1]-1,2,-1):
        encoded = keras.layers.Dense(i-1,  activation='relu')(encoded)
    
    encoded = keras.layers.Dense(1,activation='relu')(encoded)
    
    ############
    # Decoder
    ############
    decoded = keras.layers.Dense(2,  activation='relu')(encoded)

    for i in range(3,A.shape[1],1):
        decoded = keras.layers.Dense(i, activation='relu')(decoded)
    
    decoded = keras.layers.Dense(A.shape[1], activation='sigmoid')(decoded)
    
    ###############
    # Autoencoder
    ###############
    print("  Training Autoencoder... \n")
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=dP.l_rate,
            decay_steps=dP.epochs,
            decay_rate=dP.l_rdecay)
    optim2 = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9,
                beta_2=0.999, epsilon=1e-08,
                amsgrad=False)
    optim = keras.optimizers.Adam()
    
    autoencoder = keras.Model(input, decoded)
    autoencoder.compile(loss='mean_squared_error', optimizer = optim)
    history = autoencoder.fit(A, A, batch_size=dP.batch_size, epochs=dP.epochs,
        shuffle = True, verbose=1, validation_split=dP.validation_split)
        #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        
    saved_model_autoenc = os.path.splitext(file)[0]+".keras"
    print("\n  Autoencoder saved in:", saved_model_autoenc,"\n")
    autoencoder.save(saved_model_autoenc)
    
    return autoencoder

#************************************
# Generate data from Autoencoder
#************************************
def generateData(dP, autoencoder, En, A, M):
    with open(dP.norm_file, "rb") as f:
        norm = pickle.load(f)
                
    for i in range(dP.numAdditions):
        normDea = autoencoder.predict(A)
        invDea = norm.transform_inverse(normDea)
        #print("normDea", normDea)
        #print("invDea", invDea)
    
    tempM = np.vstack([En, norm.transform_inverse(M[1:,:])])
    newTrain = np.vstack([tempM, invDea])
    
    print(newTrain)
    
    return newTrain
 
#************************************
# Open Learning Data
#************************************
def readLearnFile(learnFile, newNorm):
    print(" Opening learning file: "+learnFile+"\n")
    try:
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        elif os.path.splitext(learnFile)[1] == ".h5":
            with h5py.File(learnFile, 'r') as hf:
                M = hf["M"][:]
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
    except:
        print("\033[1m" + " Learning file not found \n" + "\033[0m")
        return

    if dP.normalize:
        print("  Normalization of feature matrix to 1")
        if newNorm:
            print("  Normalization parameters saved in:", dP.norm_file,"\n")
            norm = Normalizer(M, dP)
        else:
            print("  Normalization parameters from:", dP.norm_file,"\n")
            with open(dP.norm_file, "rb") as f:
                norm = pickle.load(f)
        M = norm.transform(M)
        norm.save()

    En = M[0,:]
    A = M[1:,:]
    Cl = M[1:,0]
    
    return En, A, M

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

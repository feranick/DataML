#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* AddDenoiseAutoEncoder
* Data Augmentation via Denoising Autoencoder
* version: v2024.11.26.4
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py, pickle, configparser
from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def AddDenoiseAutoEncoder():
    main()

#************************************
# Parameters definition
#************************************
class Conf():
    def __init__(self):
        self.appName = "AddDenoiseAutoEncoder"
        confFileName = "AddDenoiseAutoEncoder.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "./"
        
        self.norm_file = self.model_directory+"norm_file.pkl"
        self.numLabels = 1
    
    def denDaeDef(self):
        self.conf['Parameters'] = {
            'saveAsTxt' : True,
            'deepAutoencoder' : True,
            'encoded_dim' : 1,
            'batch_size' : 32,
            'epochs' : 200,
            'validation_split' : 0.1,
            'regL1' : 1e-5,
            'l_rate' : 0.1,
            'l_rdecay' : 0.01,
            'min_loss_dae' : 0.025,
            'numAdditions' : 300,
            'numAddedNoisyDataBlocks' : 20,
            'percNoiseDistrMax' : 0.05,
            'excludeZeroFeatures' : True,
            'removeSpurious' : True,
            'normalize' : True,
            'normalizeLabel' : True,
            }
            
    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.denDaeDef = self.conf['Parameters']
        
            self.saveAsTxt = self.conf.getboolean('Parameters','saveAsTxt')
            self.deepAutoencoder = self.conf.getboolean('Parameters','deepAutoencoder')
            self.encoded_dim = self.conf.getint('Parameters','encoded_dim')
            self.batch_size = self.conf.getint('Parameters','batch_size')
            self.epochs = self.conf.getint('Parameters','epochs')
            self.validation_split = self.conf.getfloat('Parameters','validation_split')
            self.regL1 = self.conf.getfloat('Parameters','regL1')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.l_rdecay = self.conf.getfloat('Parameters','l_rdecay')
            self.min_loss_dae = self.conf.getfloat('Parameters','min_loss_dae')
            self.numAdditions = self.conf.getint('Parameters','numAdditions')
            self.numAddedNoisyDataBlocks = self.conf.getint('Parameters','numAddedNoisyDataBlocks')
            self.percNoiseDistrMax = self.conf.getfloat('Parameters','percNoiseDistrMax')
            self.excludeZeroFeatures = self.conf.getboolean('Parameters','excludeZeroFeatures')
            self.removeSpurious = self.conf.getboolean('Parameters','removeSpurious')
            self.normalize = self.conf.getboolean('Parameters','normalize')
            self.normalizeLabel = self.conf.getboolean('Parameters','normalizeLabel')
        except:
            print(" Error in reading configuration file. Please check it\n")

    # Create configuration file
    def createConfig(self):
        try:
            self.denDaeDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
        except:
            print("Error in creating configuration file")
    
    '''
    import tensorflow as tf
    seed_value = 10  # Choose any integer
    tf.random.set_seed(seed_value)
    '''
    
#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 AddDenoiseAutoEncoder.py <learnData>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    dP = Conf()

    En, A, M = readLearnFile(dP, sys.argv[1], True)
    
    with open(dP.norm_file, "rb") as f:
        norm = pickle.load(f)
        
    newA = norm.transform_inverse(M[1:,:])
    success = 0
    for i in range(dP.numAdditions):
        noisy_A, new_A = createNoisyData(dP, A)
        dae, val_loss = trainAutoencoder(dP, noisy_A, new_A, sys.argv[1])
        if val_loss < dP.min_loss_dae:
            A_tmp = generateData(dP, dae, En, A, M, norm)
            newA = np.vstack([newA, A_tmp])
            success += 1
            print("\n  Successful. Added so far:",str(success),"\n")
        else:
            #A_tmp = generateData(dP, dae, En, A, M, norm)
            print("  Skip this denoising autoencoder. Added so far:",str(success),"\n")
        
    if success !=0:
        if dP.removeSpurious:
            newA = removeSpurious(A, newA, norm)
            print("  Spurious data removed.")
            tag = '_noSpur'
        else:
            tag = ''
        newTrain = np.vstack([En, newA])
        print("\n  Added",str(success*A.shape[0]),"new data\n")
        newFile = dP.model_directory + os.path.splitext(os.path.basename(sys.argv[1]))[0] + '_numDataTrainDae' + \
            str(dP.numAddedNoisyDataBlocks * A.shape[0]) + '_numAdded' + str(success*A.shape[0]) + tag
        saveLearnFile(dP, newA, newFile, "")
    else:
        print("  No new training data created. Try to increse numAdditions or/and min_loss_dae.\n")

#******************************************************
# Create new Training data by adding a percentage of the max
# for that feature
#******************************************************
def getAmin(A):
    A_min = []
    for i in range(A.shape[1]):
        A_min_single = min(x for x in A[:,i] if x != 0)
        A_min = np.hstack([A_min,A_min_single])
    A_min = np.asarray(A_min)
    return A_min
        
def createNoisyData(dP, A):
    import random
    
    noisyA = np.zeros((0, A.shape[1]))
    newA = np.zeros((0, A.shape[1]))
    
    #A_min = A.min(axis=0)
    A_min = getAmin(A)
    A_max = A.max(axis=0)
    A_mean = np.mean(A, axis=0)
    A_std = A.std(axis=0)
    
    for h in range(int(dP.numAddedNoisyDataBlocks)):
        for i in range(A.shape[0]):
            noisyA_tmp = []
            A_tmp = []
            if any(A[i][1:]) != 0:
                for j in range(A.shape[1]):
                    if A[i][j] == 0 and dP.excludeZeroFeatures:
                        tmp = A[i][j]
                    else:
                        tmp =  A[i][j]+A_max[j]*(np.random.uniform(-dP.percNoiseDistrMax, dP.percNoiseDistrMax, 1))
                        if tmp<0:
                            tmp=-tmp
                        if tmp<A_min[j]:
                            tmp=0
                        
                    noisyA_tmp = np.hstack([noisyA_tmp, tmp])
                    A_tmp = np.hstack([A_tmp, A[i][j]])
                noisyA = np.vstack([noisyA, noisyA_tmp])
                newA = np.vstack([newA, A_tmp])
        #print(h,"A:",A.shape)
        #print(h,"noisyA:",noisyA.shape)
        #print(h, "newA:",newA.shape)
    return noisyA, newA

#************************************
# Train Autoencoder
#************************************
def trainAutoencoder(dP, noisyA, A, file):
    import keras
    input = keras.Input(shape=(A.shape[1],),sparse=True)
    ############
    # Encoder
    ############
    if dP.deepAutoencoder and A.shape[1] > dP.encoded_dim+2:
        encoded = keras.layers.Dense(A.shape[1]-1, activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(input)
        for i in range(A.shape[1]-1,dP.encoded_dim+1,-1):
            encoded = keras.layers.Dense(i-1,  activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
        encoded = keras.layers.Dense(dP.encoded_dim,activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
    else:
        encoded = keras.layers.Dense(dP.encoded_dim,activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(input)
    
    ############
    # Decoder
    ############
    if dP.deepAutoencoder and A.shape[1] > dP.encoded_dim+2:
        decoded = keras.layers.Dense(dP.encoded_dim+1,  activation='relu')(encoded)
        for i in range(dP.encoded_dim+2,A.shape[1],1):
            decoded = keras.layers.Dense(i, activation='relu')(decoded)
        decoded = keras.layers.Dense(A.shape[1], activation='sigmoid')(decoded)
    else:
        decoded = keras.layers.Dense(A.shape[1], activation='sigmoid')(encoded)
    
    ###############
    # Autoencoder
    ###############
    if dP.deepAutoencoder and A.shape[1] > dP.encoded_dim+2:
        print("  Training Deep Autoencoder \n   Hidden layers:",A.shape[1]-dP.encoded_dim,
            "\n   Encoded dimension:",dP.encoded_dim,"\n")
    else:
        print("  Training shallow Autoencoder \n   Encoded dimension:",dP.encoded_dim,"\n")

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
    
    log = autoencoder.fit(noisyA, A, batch_size=dP.batch_size, epochs=dP.epochs,
        shuffle = True, verbose=1, validation_split=dP.validation_split)
        #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        
    final_val_loss = np.asarray(log.history['val_loss'])[-1]
                
    saved_model_autoenc = dP.model_directory + os.path.splitext(os.path.basename(file))[0]+"_denoiseAE.keras"
    print("\n  Autoencoder saved in:", saved_model_autoenc,"\n")
    autoencoder.save(saved_model_autoenc)
    
    return autoencoder, final_val_loss

#************************************
# Generate data from Autoencoder
#************************************
def removeSpurious(A, T, norm):
    A_min = norm.transform_inverse(np.asarray([getAmin(A)]))[0]
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i,j] < A_min[j]:
                T[i,j] = 0
    return T

def generateData(dP, autoencoder, En, A, M, norm):
    #newTrain = np.vstack([En, norm.transform_inverse(M[1:,:])])
    normDea = autoencoder.predict(A)
    invDea = norm.transform_inverse(normDea)
    #print("normDea", normDea)
    #print("invDea", invDea)
    return invDea
 
#************************************
# Open Learning Data
#************************************
def readLearnFile(dP, learnFile, newNorm):
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
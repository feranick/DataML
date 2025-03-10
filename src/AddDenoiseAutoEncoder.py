#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* AddDenoiseAutoEncoder
* Data Augmentation via Denoising Autoencoder
* version: 2025.03.07.1
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
        
        self.modelName = "model_DAE.keras"
        
        self.norm_file = self.model_directory+"norm_file.pkl"
        self.numLabels = 1
            
    def denDaeDef(self):
        self.conf['Parameters'] = {
            'saveAsTxt' : True,
            'deepAutoencoder' : True,
            'reinforce' : False,
            'shuffle' : True,
            'linear_net' : False,
            'net_arch' : [12, 9, 6, 3],
            'encoded_dim' : 2,
            'batch_size' : 32,
            'epochs' : 200,
            'validation_split' : 0.1,
            'regL1' : 1e-5,
            'l_rate' : 0.1,
            'l_rdecay' : 0.01,
            'min_loss_dae' : 0.025,
            'numAdditions' : 100,
            'numAddedNoisyDataBlocks' : 100,
            'percNoiseDistrMax' : 0.025,
            'excludeZeroFeatures' : True,
            'removeSpurious' : True,
            'normalize' : True,
            'normalizeLabel' : True,
            'plotAugmData' : False,
            }
            
    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.denDaeDef = self.conf['Parameters']
        
            self.saveAsTxt = self.conf.getboolean('Parameters','saveAsTxt')
            self.deepAutoencoder = self.conf.getboolean('Parameters','deepAutoencoder')
            self.reinforce = self.conf.getboolean('Parameters','reinforce')
            self.shuffle = self.conf.getboolean('Parameters','shuffle')
            self.linear_net = self.conf.getboolean('Parameters','linear_net')
            self.net_arch = eval(self.denDaeDef['net_arch'])
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
            self.plotAugmData = self.conf.getboolean('Parameters','plotAugmData')
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
    dP = Conf()
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 AddDenoiseAutoEncoder.py <learnData>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, A, M = readLearnFile(dP, sys.argv[1], True)
    
    rootFile = dP.model_directory + os.path.splitext(os.path.basename(sys.argv[1]))[0] + \
            '_numDataTrainDae' + str(dP.numAddedNoisyDataBlocks * A.shape[0])
    
    if dP.normalize:
        with open(dP.norm_file, "rb") as f:
            norm = pickle.load(f)
        newA = norm.transform_inverse(M[1:,:])
    else:
        newA = A
        norm = 0
    
    plotAugmData(A.shape, A, rootFile+"_initial-plots.pdf")
    
    success = 0
    for i in range(dP.numAdditions):
        if dP.shuffle:
            np.random.shuffle(A)
        noisy_A, new_A = createNoisyData(dP, A)
        dae, val_loss = trainAutoencoder(dP, noisy_A, new_A, sys.argv[1])
        if val_loss < dP.min_loss_dae:
            A_tmp = generateData(dP, dae, En, A, M, norm)
            newA = np.vstack([newA, A_tmp])
            success += 1
            print("\n  Successful. Added so far:",str(success),"\n")
            #plotAugmData(A.shape, newA, rootFile+"_"+str(i)+"_plots.pdf")
        else:
            #A_tmp = generateData(dP, dae, En, A, M, norm)
            print("  Skip this denoising autoencoder. Added so far:",str(success),"\n")
        
    if success !=0:
        if dP.removeSpurious:
            newA = removeSpurious(A, newA, norm, dP)
            print("  Spurious data removed.")
            tag = '_noSpur'
        else:
            tag = ''
        newTrain = np.vstack([En, newA])
        print("\n  Added",str(success*A.shape[0]),"new data")
        newFile = rootFile + '_numAdded' + str(success*A.shape[0]) + tag
        saveLearnFile(dP, newA, newFile, "")
        
        if dP.plotAugmData:
            plotAugmData(A.shape, newA, newFile+"_plots.pdf")
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
                        tmp =  A[i][j] + A_mean[j]*(np.random.uniform(-dP.percNoiseDistrMax, dP.percNoiseDistrMax, 1))
                        if tmp<0:
                            tmp=-tmp
                        if tmp<A_min[j]:
                            tmp=0
                    noisyA_tmp = np.hstack([noisyA_tmp, tmp])
                    A_tmp = np.hstack([A_tmp, A[i][j]])
                if all(A_tmp) != 0 and all(noisyA_tmp) != 0:
                    noisyA = np.vstack([noisyA, noisyA_tmp])
                    newA = np.vstack([newA, A_tmp])

    #np.savetxt("test_newA.csv", newA, delimiter=",")
    #np.savetxt("test_noisyA.csv", noisyA, delimiter=",")
    #plotAugmData([2,4], newA, "test_newA_plots.pdf")
    #plotAugmData([2,4], noisyA, "test_noisyA_plots.pdf")
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
        if dP.linear_net:
            for i in range(A.shape[1]-1,dP.encoded_dim+1,-1):
                encoded = keras.layers.Dense(i-1,  activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
        else:
            for i in range(len(dP.net_arch)):
                encoded = keras.layers.Dense(dP.net_arch[i],  activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
        
        encoded = keras.layers.Dense(dP.encoded_dim,activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
    else:
        encoded = keras.layers.Dense(dP.encoded_dim,activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(input)
    
    ############
    # Decoder
    ############
    if dP.deepAutoencoder and A.shape[1] > dP.encoded_dim+2:
        decoded = keras.layers.Dense(dP.encoded_dim+1,  activation='relu')(encoded)
        if dP.linear_net:
            for i in range(dP.encoded_dim+2,A.shape[1],1):
                decoded = keras.layers.Dense(i, activation='relu')(decoded)
        else:
            for i in range(len(dP.net_arch)-1,-1,-1):
                decoded = keras.layers.Dense(dP.net_arch[i], activation='relu')(decoded)
        decoded = keras.layers.Dense(A.shape[1], activation='sigmoid')(decoded)
    else:
        decoded = keras.layers.Dense(A.shape[1], activation='sigmoid')(encoded)
    
    ###############
    # Autoencoder
    ###############
    if dP.deepAutoencoder and A.shape[1] > dP.encoded_dim+2:
        if dP.linear_net:
            print("\n  Training Deep Autoencoder with linear architecture\n   Hidden layers:",A.shape[1]-dP.encoded_dim,
                "\n   Encoded dimension:",dP.encoded_dim,"\n")
        else:
            print("  Training Deep Autoencoder with discrete architecture\n   Hidden layers:",dP.net_arch,
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
    
    if dP.reinforce and os.path.exists(dP.modelName):
        print("  Loading existing DAE model:",dP.modelName,"\n")
        autoencoder = keras.saving.load_model(dP.modelName)
    else:
        print("  Initializing new DAE model:",dP.modelName,"\n")
        autoencoder = keras.Model(input, decoded)
        autoencoder.compile(loss='mean_squared_error', optimizer = optim)
    
    log = autoencoder.fit(noisyA, A, batch_size=dP.batch_size, epochs=dP.epochs,
        shuffle = True, verbose=1, validation_split=dP.validation_split)
        #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        
    final_val_loss = np.asarray(log.history['val_loss'])[-1]
                
    print("\n  Autoencoder saved in:", dP.modelName,"\n")
    autoencoder.save(dP.modelName)
    
    return autoencoder, final_val_loss

#************************************
# Generate data from Autoencoder
#************************************
def removeSpurious(A, T, norm, dP):
    if dP.normalize:
        A_min = norm.transform_inverse(np.asarray([getAmin(A)]))[0]
    else:
        A_min = getAmin(A)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i,j] < A_min[j]:
                T[i,j] = 0
    return T

def generateData(dP, autoencoder, En, A, M, norm):
    #newTrain = np.vstack([En, norm.transform_inverse(M[1:,:])])
    normDea = autoencoder.predict(A)
    if dP.normalize:
        normDea = norm.transform_inverse(normDea)
    #print("normDea", normDea)
    #print("invDea", invDea)
    return normDea
 
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

    if dP.excludeZeroFeatures:
        ind = np.any(M == 0, axis=1)
        M = M[~ind]
    
    En = M[0,:]
    A = M[1:,:]
    Cl = M[1:,0]
    return En, A, M

#************************************
# Plot augmented training data
#************************************
def plotAugmData(shape, newA, plotFile):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf = PdfPages(plotFile)
    
    for i in range(1, shape[1]):
        x = newA[:shape[0],0]
        y = newA[:shape[0]:,i]
        xA = newA[shape[0]:,0]
        yA = newA[shape[0]:,i]
        plt.plot(xA,yA, 'bo', markersize=3)
        plt.plot(x,y, 'ro', markersize=3)
        plt.xlabel("col "+str(i))
        plt.ylabel("col 0")
        pdf.savefig()
        plt.close()
    pdf.close()
    
#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

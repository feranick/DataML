#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* DataML_DAE
* Generative AI via Denoising Autoencoder
* version: 2025.12.11.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py, pickle, configparser, ast, getopt
from numpy.polynomial.polynomial import Polynomial as polyfit
from numpy.polynomial.polynomial import polyval as polyval

from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def DataML_DAE():
    main()

#************************************
# Parameters definition
#************************************
class Conf():
    def __init__(self):
    
        #################################
        ### Types of noise:
        ### Set using: typeNoise
        ### - Random (default)
        ### - RandomYFit
        ### - RandomXFit
        ### - ColumnValueSwap
        #################################
        
        self.appName = "DataML_DAE"
        confFileName = "DataML_DAE.ini"
        self.configFile = os.path.join(os.getcwd(),confFileName)
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "./"
        
        self.modelName = "model_DAE.keras"
        
        self.norm_file = self.model_directory+"norm_file_DAE.pkl"
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
            'typeNoise' : 'Random',
            'fitPolyDegree' : 3,
            'numColSwaps' : 10,
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
            self.denDaePar = self.conf['Parameters']
        
            self.saveAsTxt = self.conf.getboolean('Parameters','saveAsTxt')
            self.deepAutoencoder = self.conf.getboolean('Parameters','deepAutoencoder')
            self.reinforce = self.conf.getboolean('Parameters','reinforce')
            self.shuffle = self.conf.getboolean('Parameters','shuffle')
            self.linear_net = self.conf.getboolean('Parameters','linear_net')
            self.net_arch = ast.literal_eval(self.denDaePar['net_arch'])
            self.encoded_dim = self.conf.getint('Parameters','encoded_dim')
            self.batch_size = self.conf.getint('Parameters','batch_size')
            self.epochs = self.conf.getint('Parameters','epochs')
            self.validation_split = self.conf.getfloat('Parameters','validation_split')
            self.regL1 = self.conf.getfloat('Parameters','regL1')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.l_rdecay = self.conf.getfloat('Parameters','l_rdecay')
            self.typeNoise = self.conf.get('Parameters','typeNoise')
            self.fitPolyDegree = self.conf.getint('Parameters','fitPolyDegree')
            self.numColSwaps = self.conf.getint('Parameters','numColSwaps')
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
    
    # update configuration file
    def updateConfig(self, section, par, value):
        if self.conf.has_option(section, par) is True:
            self.conf.set(section, par, value)
            try:
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
    try:
        opts, args = getopt.getopt(sys.argv[1:],
            "tag:", ["train", "augment", "generate"])
    except:
        usage()
        sys.exit(2)
    
    if opts == []:
        usage()
        sys.exit(2)
        
    for o, a in opts:
        if o in ("-t" , "--train"):
            try:
                train(sys.argv[2])
            except:
                usage()
                sys.exit(2)
                
        if o in ("-a" , "--augment"):
            try:
                augment(sys.argv[2], True)
            except:
                usage()
                sys.exit(2)
        
        if o in ("-g" , "--generate"):
            try:
                generate(sys.argv[2])
            except:
                usage()
                sys.exit(2)
        
#***********************************************
# Train DAE via final learning file
#***********************************************
def train(learnFile):
    dP = Conf()
    dP.updateConfig('Parameters','reinforce','True')
    old_numAdditions = dP.numAdditions
    dP.updateConfig('Parameters','numAdditions','1')
    augment(learnFile, False)
    dP.updateConfig('Parameters','reinforce','False')
    dP.updateConfig('Parameters','numAdditions',str(old_numAdditions))
    
#***********************************************
# Generate new sample based on prompt
#***********************************************
def generate(csvFile):
    import keras
    import pandas as pd
    from datetime import datetime, date
    dP = Conf()
    
    print(f"  Opening file with prompt samples: {csvFile}")
    dataDf = pd.read_csv(csvFile)
    
    
    newDataDf = pd.DataFrame(dataDf[dataDf.columns[0]])
        
    print("  Loading existing DAE model:",dP.modelName,"\n")
    autoencoder = keras.saving.load_model(dP.modelName)
    
    if dP.normalize:
        try:
            with open(dP.norm_file, "rb") as f:
                norm = pickle.load(f)
            print("  Opening pkl file with normalization data:",dP.norm_file)
        except:
            print("\033[1m pkl file not found \033[0m")
            sys.exit()
    else:
        norm = None
    
    for i in range(1,dataDf.shape[1]):
        R = np.array([dataDf.iloc[:,i].tolist()], dtype=float)
        Rorig = np.copy(R)
        
        if dP.normalize:
            R = norm.transform_valid_data_DAE(R)
        
        newR = autoencoder.predict(R)
        #print("\nThis is the predicted R:",newR)
        if dP.normalize:
            newR = norm.transform_inverse(newR)
        newDataDf[dataDf.columns[i]] = newR.flatten()
    
    print('\n  ==============================================================================')
    print('  \033[1m Generated data\033[0m')
    print('  ==============================================================================')
    for i in range(1,dataDf.shape[1]):
        tmp = pd.DataFrame(dataDf[dataDf.columns[0]])
        tmp.rename(columns={ dataDf.columns[0]: dataDf.columns[i]}, inplace=True)
        print('  --------------------------------------------------------------------------------')
        tmp["input"] = dataDf[dataDf.columns[i]]
        tmp["output"] = newDataDf[newDataDf.columns[i]]
        print(tmp)
    print('  --------------------------------------------------------------------------------\n')
    
    summaryCSVFileRoot = os.path.splitext(csvFile)[0]
    summaryCSVFile = summaryCSVFileRoot+"_DAE_output"+str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))
    
    newDataDf.rename(columns={ dataDf.columns[0]: "DAE output"}, inplace=True)
    newDataDf.to_csv(summaryCSVFile, index=False, sep=',')
    
    print(f" DAE generated samples saved in: {summaryCSVFile}\n")
        
#***********************************************
# Augment learning data via DAE
# Create a new learning file with added data
#***********************************************
def augment(learnFile,augFlag):
    dP = Conf()
    try:
        En, A, M = readLearnFileDAE(learnFile, True, dP)
    except:
        return 1
    
    rootFile = dP.model_directory + os.path.splitext(os.path.basename(learnFile))[0] + \
            '_numDataTrainDae' + str(dP.numAddedNoisyDataBlocks * A.shape[0])
    
    if dP.normalize:
        with open(dP.norm_file, "rb") as f:
            norm = pickle.load(f)
        newA = norm.transform_inverse(M[1:,:])
    else:
        newA = A
        norm = 0
    
    if dP.plotAugmData:
        plotData(dP, A, None, True, True, "Initial (X-F) Data", rootFile+"_initial_X-F_plots.pdf")
        plotData(dP, A, None, False, True, "Initial (Y-P) Data", rootFile+"_initial_Y-P_plots.pdf")
    
    success = 0
    plotFeatType = True
    for i in range(dP.numAdditions):
        if dP.shuffle:
            np.random.shuffle(A)
        if dP.typeNoise == 'Random':
            print("  Creating random noise. \n")
            noisy_A, new_A = createNoisyData(dP, A)
        elif dP.typeNoise == 'RandomYFit':
            print("  Creating random noise from fitted initial feature data.\n")
            noisy_A, new_A = createYFitNoisyData(dP, A)
        elif dP.typeNoise == 'RandomXFit':
            print("  Creating random noise from fitted initial predicted data.\n")
            noisy_A, new_A = createXFitNoisyData(dP, A)
            plotFeatType = False
        elif dP.typeNoise == 'ColumnValueSwap':
            print("  Creating noise via random swapping of values within columns.\n")
            noisy_A, new_A = swapValuesColumn(dP, A)
        else:
            print("  Check value of typeNoise in ini file. Aborting.")
            return 0;
            
        dae, val_loss = trainAutoencoder(dP, noisy_A, new_A, sys.argv[1])
        
        if augFlag:
            if val_loss < dP.min_loss_dae:
                A_tmp = generateData(dP, dae, En, A, M, norm)
                newA = np.vstack([newA, A_tmp])
                success += 1
                print("\n  Successful. Added so far:",str(success),"\n")
                #plotData(dP, A, newA, plotFeatType, "test", True, rootFile+"_"+str(i)+"_plots.pdf")
            else:
                #A_tmp = generateData(dP, dae, En, A, M, norm)
                print("  Skip this denoising autoencoder. Added so far:",str(success),"\n")
        else:
            print("  Trained DAE model\n")
            return
        
    if success !=0:
        tag = "_"+dP.typeNoise
        if dP.removeSpurious:
            newA = removeSpurious(A, newA, norm, dP)
            print("  Spurious data removed.")
            tag += '_noSpur'
        newTrain = np.vstack([En, newA])
        print("\n  Added",str(success*A.shape[0]),"new data")
        newFile = rootFile + '_numAdded' + str(success*A.shape[0]) + tag
        saveLearnFile(dP, newA, newFile, "")
        
        if dP.plotAugmData:
            plotData(dP, A, newA, plotFeatType, False, "Augmented data", newFile+"_plots.pdf")
    else:
        print("  No new training data created. Try to increse numAdditions or/and min_loss_dae.\n")

#******************************************************
# Noise generation
#******************************************************
def getAmin(A):
    A_min = []
    for i in range(A.shape[1]):
        A_min_single = min(x for x in A[:,i] if x != 0)
        A_min.append(A_min_single)
    return np.hstack(A_min)
    
# ---------------------------------
# Create new Training data
# by adding a random percentage
# of the mean for that feature
# ---------------------------------
def createNoisyData(dP, A):
    import random
        
    #A_min = A.min(axis=0)
    A_min = getAmin(A)
    #A_max = A.max(axis=0)
    A_mean = np.mean(A, axis=0)
    #A_std = A.std(axis=0)
    
    noisyA_list = []
    newA_list = []

    for _ in range(int(dP.numAddedNoisyDataBlocks)):
        for i in range(A.shape[0]):
            if np.any(A[i, 1:] != 0):  # Efficient zero check
                noisyA_row = np.zeros(A.shape[1])
                newA_row = np.copy(A[i]) # copy the row.

                for j in range(A.shape[1]):
                    if A[i, j] == 0 and dP.excludeZeroFeatures:
                        noisyA_row[j] = A[i, j]
                    else:
                        noisyA_row[j] = A[i, j] + A_mean[j] * np.random.uniform(-dP.percNoiseDistrMax, dP.percNoiseDistrMax)
                        if noisyA_row[j] < 0:
                            noisyA_row[j] = -noisyA_row[j]
                        if noisyA_row[j] < A_min[j]:
                            noisyA_row[j] = 0

                if np.all(newA_row != 0) and np.all(noisyA_row != 0):
                    noisyA_list.append(noisyA_row)
                    newA_list.append(newA_row)

    noisyA = np.vstack(noisyA_list) if noisyA_list else np.empty((0, A.shape[1]))
    newA = np.vstack(newA_list) if newA_list else np.empty((0, A.shape[1]))
    
    plotData(dP, A, noisyA, True, True, "Noisy", "Noisy.pdf")
    return noisyA, newA

# ------------------------------------
# Create new Training data
# by adding a random percentage
# to the fitted featured initial data
# ------------------------------------
def createYFitNoisyData(dP, A):
    import random
    poly = fitReverseInitialData(dP, A)
    noisyA_list = []
    newA_list = []
    for h in range(int(dP.numAddedNoisyDataBlocks)):
        x_tmp = A[:,0] + np.random.uniform(-dP.percNoiseDistrMax, dP.percNoiseDistrMax, A.shape[0])
        nA_tmp = [x_tmp.reshape(-1,1)]
        for j in range(1,A.shape[1]):
            tmp = (polyval(x_tmp, poly[j]) + np.random.uniform(-dP.percNoiseDistrMax, dP.percNoiseDistrMax, A.shape[0])).reshape(-1,1)
            #tmp = (polyval(A[:,0], poly[j])).reshape(-1,1)
            nA_tmp.append(tmp)
        noisyA_list.append(np.hstack(nA_tmp))
        newA_list.append(A)
        
    noisyA = np.vstack(noisyA_list)
    newA = np.vstack(newA_list)
        
    plotData(dP, A, noisyA, False, True, "NoisyY", "NoisyY.pdf")
    return noisyA, newA
    
# Fit initial data from prediction vs features
def fitReverseInitialData(dP, A):
    poly = np.zeros((A.shape[1], int(dP.fitPolyDegree)+1))
    for i in range(A.shape[1]):
        poly[i, :] = polyfit.fit(A[:, 0], A[:, i], int(dP.fitPolyDegree)).coef
    return poly

# --------------------------------------
# Create new Training data
# by adding a random percentage
# to the fitted predictedd initial data
# --------------------------------------
def createXFitNoisyData(dP, A):
    import random
    poly = fitInitialData(dP, A)
    noisyA_list = []
    newA_list = []
    for h in range(int(dP.numAddedNoisyDataBlocks)):
        x_tmp_list = []
        y_tmp_list = []

        for j in range(1, A.shape[1]):
            x_tmp = (A[:, j] + np.random.uniform(-dP.percNoiseDistrMax, dP.percNoiseDistrMax, A.shape[0])).reshape(-1, 1)
            y_tmp_list.append((polyval(x_tmp, poly[j])).reshape(-1, 1))
            x_tmp_list.append(x_tmp)

        y_tmp = np.hstack(y_tmp_list)
        noisyA_list.append(np.hstack([np.mean(y_tmp, axis=1).reshape(-1, 1), np.hstack(x_tmp_list)]))
        newA_list.append(A)

    noisyA = np.vstack(noisyA_list)
    newA = np.vstack(newA_list)

    plotAugmData(dP, A, noisyA, True, True, "NoisyX", "NoisyX.pdf")
    return noisyA, newA
    
# Fit initial data from features vs prediction
def fitInitialData(dP, A):
    poly = np.zeros((A.shape[1], int(dP.fitPolyDegree) + 1))
    for i in range(A.shape[1]):
        poly[i, :] = polyfit.fit(A[:, i], A[:, 0], int(dP.fitPolyDegree)).coef
    return poly

# ---------------------------------
# Create new Training data
# by swapping 2 values
# within the same column
# ---------------------------------
def swapValuesColumn(dP, A):
    import random
    rows, cols = A.shape
    if rows < 2:
        print("Warning: The array has less than 2 rows. No swaps can be performed.")
        return A, A
        
    noisyA_list = []
    newA_list = []
    for _ in range(int(dP.numAddedNoisyDataBlocks)):
        noisyA_tmp = np.copy(A)
        for _ in range(int(dP.numColSwaps)):
            col_index = random.randint(0, cols - 1)
            row_index1, row_index2 = random.sample(range(rows), 2)
            noisyA_tmp[row_index1, col_index], noisyA_tmp[row_index2, col_index] = A[row_index2, col_index], A[row_index1, col_index]

        noisyA_list.append(noisyA_tmp)
        newA_list.append(A)
    noisyA = np.vstack(noisyA_list)
    newA = np.vstack(newA_list)
    plotAugmData(dP, A.shape, noisyA, True, True, "Swap", "Swap.pdf")
    return noisyA, newA
    

#************************************
# Train Autoencoder
#************************************
def trainAutoencoder(dP, noisyA, A, file):
    import keras
    #input = keras.Input(shape=(A.shape[1],),sparse=True)
    input = keras.Input(shape=(A.shape[1],))
    ############
    # Encoder
    ############
    encoded = keras.layers.Dense(A.shape[1]-1, activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(input)
    if dP.deepAutoencoder:
        if dP.linear_net:
            if A.shape[1] > dP.encoded_dim+2:
                for i in range(A.shape[1]-1,dP.encoded_dim+1,-1):
                    encoded = keras.layers.Dense(i-1,  activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
                encoded = keras.layers.Dense(dP.encoded_dim,activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
        else:
            for i in range(len(dP.net_arch)):
                encoded = keras.layers.Dense(dP.net_arch[i],  activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
            encoded = keras.layers.Dense(dP.encoded_dim,activation='relu',activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
    
    ############
    # Decoder
    ############
    if dP.deepAutoencoder:
        decoded = keras.layers.Dense(dP.encoded_dim+1,  activation='relu')(encoded)
        if dP.linear_net:
            if A.shape[1] > dP.encoded_dim+2:
                for i in range(dP.encoded_dim+2,A.shape[1],1):
                    decoded = keras.layers.Dense(i, activation='relu')(decoded)
            else:
                decoded = keras.layers.Dense(A.shape[1], activation='sigmoid')(encoded)
        else:
            for i in range(len(dP.net_arch)-1,-1,-1):
                decoded = keras.layers.Dense(dP.net_arch[i], activation='relu')(decoded)
        decoded = keras.layers.Dense(A.shape[1], activation='sigmoid')(decoded)
    else:
        decoded = keras.layers.Dense(A.shape[1], activation='sigmoid')(encoded)
    
    ###############
    # Autoencoder
    ###############
    if dP.deepAutoencoder:
        if dP.linear_net:
            if A.shape[1] > dP.encoded_dim+2:
                print("\n  Training Deep Autoencoder with linear architecture\n   Hidden layers:",A.shape[1]-dP.encoded_dim,
                    "\n   Encoded dimension:",dP.encoded_dim,"\n")
            else:
                print("\n  Training shallow Autoencoder \n   Encoded dimension:",dP.encoded_dim,"\n")
        else:
            print("\n  Training Deep Autoencoder with discrete architecture\n   Hidden layers:",dP.net_arch,
                "\n   Encoded dimension:",dP.encoded_dim,"\n")
    else:
        print("\n  Training shallow Autoencoder \n   Encoded dimension:",dP.encoded_dim,"\n")

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
def readLearnFileDAE(learnFile, newNorm, dP):
    M = readFile(learnFile)
    
    if dP.normalize:
        print("  Normalization of feature matrix to 1")
        if newNorm:
            print("  Normalization parameters saved in:", dP.norm_file,"\n")
            norm = Normalizer(M, dP)
            norm.save()
        else:
            print("  Normalization parameters from:", dP.norm_file,"\n")
            with open(dP.norm_file, "rb") as f:
                norm = pickle.load(f)
        M = norm.transform(M)


    if dP.excludeZeroFeatures:
        ind = np.any(M == 0, axis=1)
        ind[0] = False
        M = M[~ind]

    En = M[0,:]
    A = M[1:,:]
    Cl = M[1:,0]
    
    return En, A, M

#************************************
# Plot augmented training data
#************************************
def plotData(dP, A, newA, feat, normFlag, title, plotFile):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import r2_score
    
    pdf = PdfPages(plotFile)
        
    if dP.normalize and normFlag and newA is not None:
        with open(dP.norm_file, "rb") as f:
            norm = pickle.load(f)
        newA = norm.transform_inverse(newA)
        A = norm.transform_inverse(A)
        
    for i in range(1, A.shape[1]):
        if feat:
            x = A[:,i]
            y = A[:,0]
            if newA is not None:
                xA = newA[:,i]
                yA = newA[:,0]
            plt.xlabel("col "+str(i)+" - feature parameter")
            plt.ylabel("col 0 - predicted parameter")
        else:
            y = A[:,i]
            x = A[:,0]
            if newA is not None:
                yA = newA[:,i]
                xA = newA[:,0]
            plt.xlabel("col 0 - predicted parameter")
            plt.ylabel("col "+str(i)+" - feature parameter")
        
        plt.plot(x,y, 'ro', markersize=3)
        if newA is not None:
            plt.plot(xA,yA, 'bo', markersize=3)
        poly = polyfit.fit(x, y, dP.fitPolyDegree)
        plt.plot(np.unique(x), poly(np.unique(x)))
        plt.title(title+" - $R^2={0:.3f}$".format(r2_score(y, poly(x))))
        pdf.savefig()
        plt.close()
    pdf.close()

#************************************
# Lists the program usage
#************************************
def usage():
    print('\n Usage:\n')
    print(' Train DAE:')
    print('  DataML_DAE -t <learningFile>\n')
    print(' Augment data from <learningFile> using DAE:')
    print('  DataML_DAE -a <learningFile>\n')
    print(' Generate new DAE samples from csv of incomplete samples')
    print('  DataML_DAE -g <csvlist>\n')
    
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

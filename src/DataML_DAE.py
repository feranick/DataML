#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* DataML_DAE
* Generative AI via Denoising Autoencoder
* Device config via shared DataML_Backend (TF / PyTorch / JAX), SLURM-aware
* version: 2026.9.1.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import os
import sys, os.path, h5py, pickle, configparser, ast, getopt
from numpy.polynomial.polynomial import Polynomial as polyfit
from numpy.polynomial.polynomial import polyval as polyval

# Backend-agnostic device/GPU selection lives in the shared DataML_Backend
# module. Importing it (before any keras import) performs device SELECTION;
# configureDevices() below applies the backend-specific memory tuning.
from DataML_Backend import configureDevices

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
    
        #####################################################
        ### Types of noise:
        ### Set using: typeNoise
        ### - Random (default)
        ### - RandomYFit
        ### - RandomXFit
        ### - ColumnValueSwap
        ###
        ### Activation functions:
        ### Outer layers
        ### Set using: activation
        ### - linear
        ### - sigmoid
        ### Inner activation (default: elu)
        ### Set using: innerActivation
        ### - elu
        ### - relu
        ### Loss Metric:
        ### - mean_squared_error (default, for ordered data)
        ### - mean_absolute_error (for scattered data)
        #####################################################
        
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
        self.tb_directory = "tb_model"
        
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
            'net_arch' : [20, 16, 12],
            'encoded_dim' : 8,
            'batch_size' : 16,
            'epochs' : 200,
            'validation_split' : 0.1,
            'regL1' : 0,
            'dropout' : 0,
            'l_rate' : 0.001,
            'l_rdecay' : 0.9,
            'activation' : 'linear',
            'lossMetric' : 'mean_squared_error',
            'innerActivation' : 'elu',
            'typeNoise' : 'Random',
            'fitPolyDegree' : 3,
            'numColSwaps' : 10,
            'min_loss_dae' : 0.05,
            'numAdditions' : 20,
            'numAddedNoisyDataBlocks' : 20,
            'percNoiseDistrMax' : 0.075,
            'postGenerationNoise' : True,
            'postGenerationNoiseMax' : 0.075,
            'excludeZeroFeatures' : False,
            'excludeZeroLabels' : True,
            'removeSpurious' : True,
            'normalize' : True,
            'normalizeLabel' : True,
            'discreteThreshold' : 5,
            'plotAugmData' : False,
            'stopAtBest' : False,
            'saveBestModel' : False,
            'metricBestModel' : 'val_mae',
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
            self.dropout = self.conf.getfloat('Parameters','dropout')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.l_rdecay = self.conf.getfloat('Parameters','l_rdecay')
            self.activation = self.conf.get('Parameters','activation')
            self.lossMetric = self.conf.get('Parameters','lossMetric', fallback='mean_squared_error')
            self.innerActivation = self.conf.get('Parameters','innerActivation', fallback='elu')
            self.typeNoise = self.conf.get('Parameters','typeNoise')
            self.fitPolyDegree = self.conf.getint('Parameters','fitPolyDegree')
            self.numColSwaps = self.conf.getint('Parameters','numColSwaps')
            self.min_loss_dae = self.conf.getfloat('Parameters','min_loss_dae')
            self.numAdditions = self.conf.getint('Parameters','numAdditions')
            self.numAddedNoisyDataBlocks = self.conf.getint('Parameters','numAddedNoisyDataBlocks')
            self.percNoiseDistrMax = self.conf.getfloat('Parameters','percNoiseDistrMax')
            self.postGenerationNoise = self.conf.getboolean('Parameters','postGenerationNoise')
            self.postGenerationNoiseMax = self.conf.getfloat('Parameters','postGenerationNoiseMax')
            self.excludeZeroFeatures = self.conf.getboolean('Parameters','excludeZeroFeatures')
            self.excludeZeroLabels = self.conf.getboolean('Parameters','excludeZeroLabels')
            self.removeSpurious = self.conf.getboolean('Parameters','removeSpurious')
            self.normalize = self.conf.getboolean('Parameters','normalize')
            self.normalizeLabel = self.conf.getboolean('Parameters','normalizeLabel')
            self.discreteThreshold = self.conf.getint('Parameters','discreteThreshold', fallback=5)
            self.plotAugmData = self.conf.getboolean('Parameters','plotAugmData')
            self.stopAtBest = self.conf.getboolean('Parameters','stopAtBest')
            self.saveBestModel = self.conf.getboolean('Parameters','saveBestModel')
            self.metricBestModel = self.conf.get('Parameters','metricBestModel')
        
        except Exception as e:
            print(" Error in reading configuration file:")
            print(f"  {e}\n")

    # Create configuration file
    def createConfig(self):
        try:
            self.denDaeDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
                
        except Exception as e:
            print("Error in creating configuration file:")
            print(f"  {e}\n")
    
    # update configuration file
    def updateConfig(self, section, par, value):
        if self.conf.has_option(section, par) is True:
            self.conf.set(section, par, value)
            try:
                with open(self.configFile, 'w') as configfile:
                    self.conf.write(configfile)
                    
            except Exception as e:
                print("Error in updating configuration file:")
                print(f"  {e}\n")
    
    '''
    import tensorflow as tf
    seed_value = 10  # Choose any integer
    tf.random.set_seed(seed_value)
    '''
    
#************************************
# Main
#************************************
def main():
    configureDevices()
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
            except Exception as e:
                print(f" An error occurred: {e}\n")
                sys.exit(2)
                
        if o in ("-a" , "--augment"):
            try:
                augment(sys.argv[2], True)
            except Exception as e:
                print(f" An error occurred: {e}\n")
                sys.exit(2)
        
        if o in ("-g" , "--generate"):
            try:
                generate(sys.argv[2])
            except Exception as e:
                print(f" An error occurred: {e}\n")
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
def generate(learnFile):
    """Augment learnFile with the already-trained DAE, with diagnostics.

    Behavioural changes relative to the previous version, both aimed at making
    the zeros failure impossible to miss:

      1. removeSpurious is applied under a policy. 'drop' discards offending
         rows instead of writing 0 into cells, which is what you want here --
         zeroing a cell in a column whose legitimate minimum is 122 does not
         mark it as spurious, it marks it as missing, and downstream code reads
         0 as missing.
      2. If a stage eliminates essentially everything, generate() reports the
         diagnosis and refuses to write the file.
    """
    import keras
    import numpy as np
    import os
    import pickle
    from datetime import datetime

    dP = Conf()
    policy = 'drop' # options: 'drop' | 'zero' | 'clip'

    if not os.path.exists(dP.modelName):
        print(f"\033[1m  No trained model found: {dP.modelName}\033[0m")
        print("  Train one first:  DataML_DAE -t <learningFile>\n")
        return 1

    print(f"  Opening file with seed samples: {learnFile}")
    M_raw = readFile(learnFile)
    En_orig = M_raw[0, :]

    En, A, M, empty = readLearnFileDAE(learnFile, False, dP)
    if empty:
        print("  No usable rows in the input file.\n")
        return 1

    print(f"  Loading existing DAE model: {dP.modelName}\n")
    autoencoder = keras.saving.load_model(dP.modelName)

    if dP.normalize:
        with open(dP.norm_file, "rb") as f:
            norm = pickle.load(f)
        print(f"  Normalization data from: {dP.norm_file}")
        orig_physical_A = norm.transform_inverse(A)
    else:
        norm = 0
        orig_physical_A = A

    if A.shape[1] != M_raw.shape[1]:
        print(f"\033[1m  Column count mismatch: model space has {A.shape[1]}, "
              f"file has {M_raw.shape[1]}\033[0m\n")
        return 1

    A_min = getAmin(orig_physical_A)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    blocks = []
    total = 0
    for g in range(dP.numAdditions):
        print(f"\n  Generation pass {g + 1}/{dP.numAdditions}")

        A_tmp = generateData(dP, autoencoder, A, M, norm)

        A_tmp = snap_discrete_features(orig_physical_A, A_tmp,
                                       dP.discreteThreshold)
        if A_tmp.shape[0] == 0:
            print("  All generated rows were purged by snap_discrete_features.")
            continue

        blocks.append(A_tmp)
        total += A_tmp.shape[0]
        print(f"  Kept {A_tmp.shape[0]} rows (running total {total})")

    if not blocks:
        print("\n  No usable rows generated.\n")
        return 1

    newA = np.vstack(blocks)

    # ------------------------------------------------------------------
    # Spurious handling, with a policy and a guard
    # ------------------------------------------------------------------
    if dP.removeSpurious:
        before = newA.shape[0]
        offending = np.any(newA < A_min, axis=1)
        frac_cells = float((newA < A_min).mean())

        if policy == 'drop':
            newA = newA[~offending]
            print(f"\n  Spurious rows dropped: {before - newA.shape[0]}"
                  f" of {before}")
        elif policy == 'clip':
            newA = np.maximum(newA, A_min)
            print(f"\n  Spurious values clipped up to the column minimum")
        else:
            newA = removeSpurious(orig_physical_A, newA, dP)
            print("\n  Spurious data removed (zeroed).")

        if newA.shape[0] == 0 or frac_cells > 0.90:
            print('\n  ==============================================')
            print('   \033[1mABORTED - generated data is unusable\033[0m')
            print('  ==============================================')
            print(f"  {frac_cells:.0%} of generated cells fell below the"
                  f" physical column minima.")
            print("  Nothing was written.")
            print("  removeSpurious = False to inspect the raw generated"
                  " values.")
            print("  See the CAUSE A / CAUSE B notes at the top of this file.\n")
            return 1

    newA = np.vstack([orig_physical_A, newA])
    newTrain = np.vstack([En_orig, newA])

    stamp = datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
    rootFile = (dP.model_directory
                + os.path.splitext(os.path.basename(learnFile))[0]
                + '_DAEgen_numAdded' + str(newA.shape[0]
                                           - orig_physical_A.shape[0]) + stamp)

    saveLearnFile(dP, newTrain, rootFile, "")

    n_synth = newA.shape[0] - orig_physical_A.shape[0]
    print('\n  ==============================================================')
    print('   \033[1mGenerated data\033[0m')
    print('  ==============================================================')
    print(f"  Seed rows       : {orig_physical_A.shape[0]}")
    print(f"  Generated       : {total}")
    print(f"  Kept            : {n_synth}")
    print(f"  Rows written    : {newA.shape[0]}  (real + synthetic)")
    print(f"  Model           : {dP.modelName}  (not retrained)")
    print('  ==============================================================\n')

    if dP.plotAugmData:
        plotData(dP, orig_physical_A, newA, True, False,
                 "DAE generated data", rootFile + "_plots.pdf")

    return 0
        
#***********************************************
# Augment learning data via DAE
# Create a new learning file with added data
#***********************************************
def augment(learnFile,augFlag):
    dP = Conf()
    
    # Read the raw header before any normalization
    M_raw = readFile(learnFile)
    En_orig = M_raw[0, :]
    
    try:
        En, A, M, empty = readLearnFileDAE(learnFile, True, dP)
        if empty:
            return 1
    except Exception as e:
        print(f" An error occurred during augmentation: {e}\n")
        return 1
    
    rootFile = dP.model_directory + os.path.splitext(os.path.basename(learnFile))[0] + \
            '_numDataTrainDae' + str(dP.numAddedNoisyDataBlocks * A.shape[0])
    
    if dP.normalize:
        with open(dP.norm_file, "rb") as f:
            norm = pickle.load(f)
        newA = norm.transform_inverse(M[1:,:])
        orig_physical_A = norm.transform_inverse(A)
    else:
        newA = A
        orig_physical_A = A
        norm = 0
    
    if dP.plotAugmData:
        plotData(dP, A, None, True, True, "Initial (X-F) Data", rootFile+"_initial_X-F_plots.pdf")
        plotData(dP, A, None, False, True, "Initial (Y-P) Data", rootFile+"_initial_Y-P_plots.pdf")
    
    success = 0
    total_added_rows = 0  # Tracks the exact number of rows generated
    plotFeatType = True
    for i in range(dP.numAdditions):
        print(f"\n Augmentation in progress: step {i+1}/{dP.numAdditions}\n")
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
                A_tmp = generateData(dP, dae, A, M, norm)
                
                # Snap generated discrete data & conditionally filter labels
                A_tmp = snap_discrete_features(orig_physical_A, A_tmp, dP.discreteThreshold)
                
                newA = np.vstack([newA, A_tmp])
                success += 1
                total_added_rows += A_tmp.shape[0]  # UPDATE TRACKER
                print("\n  Successful. Added blocks so far:", str(success), "| Total new rows:", str(total_added_rows), "\n")
            else:
                print("  Skip this denoising autoencoder. Added blocks so far:", str(success), "\n")
        else:
            print("  Trained DAE model\n")
            return
        
        
    printParam(dP)
    if success !=0:
        tag = "_"+dP.typeNoise
        if dP.removeSpurious:
            # Pass physical space matrices directly to avoid floating point truncation issues
            newA = removeSpurious(orig_physical_A, newA, dP)
            print("  Spurious data removed.")
            tag += '_noSpur'
            
        newTrain = np.vstack([En_orig, newA])
        
        # USE total_added_rows FOR PRINTS AND FILENAMES
        print("\n  Added", str(total_added_rows), "new data points")
        newFile = rootFile + '_numAdded' + str(total_added_rows) + tag
        
        # Use newTrain to preserve header row for DataML_DF compatibility
        saveLearnFile(dP, newTrain, newFile, "")
        
        if dP.plotAugmData:
            if dP.normalize:
                A_plot = norm.transform_inverse(A)
            else:
                A_plot = A
            plotData(dP, A_plot, newA, plotFeatType, False, "Augmented data", newFile+"_plots.pdf")
    else:
        print("  No new training data created. Try to increse numAdditions or/and min_loss_dae.\n")


#******************************************************
# Noise generation
#******************************************************
def getAmin(A):
    A_min = []
    for i in range(A.shape[1]):
        # Prevent crash if column is entirely zeros
        non_zero = A[:, i][A[:, i] != 0]
        A_min_single = non_zero.min() if non_zero.size > 0 else 0.0
        A_min.append(A_min_single)
    return np.array(A_min)
    
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

                if dP.excludeZeroFeatures:
                    if np.all(newA_row != 0) and np.all(noisyA_row != 0):
                        noisyA_list.append(noisyA_row)
                        newA_list.append(newA_row)
                else:
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
            #x_tmp_single = A[:, j] + np.random.uniform(-dP.percNoiseDistrMax, dP.percNoiseDistrMax, A.shape[0]).reshape(-1, 1)
            x_tmp_single = A[:, j].reshape(-1, 1)
            y_tmp_single = polyval(x_tmp_single, poly[j]).reshape(-1, 1)
            x_tmp_list.append(x_tmp_single)
            y_tmp_list.append(y_tmp_single)

        x_tmp = np.hstack(x_tmp_list)
        y_tmp = np.hstack(y_tmp_list)
        
        noisyA_tmp = np.hstack([np.mean(y_tmp, axis=1).reshape(-1, 1), x_tmp])
        #noisyA_tmp = np.hstack([y_tmp[:,0].reshape(-1, 1),x_tmp])
        noisyA_list.append(noisyA_tmp)
        newA_list.append(A)

    noisyA = np.vstack(noisyA_list)
    newA = np.vstack(newA_list)

    plotData(dP, A, noisyA, True, True, "NoisyX", "NoisyX.pdf")
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
    plotData(dP, A, noisyA, True, True, "Swap", "Swap.pdf")
    return noisyA, newA
    

#************************************
# Train Autoencoder
#************************************
def trainAutoencoder(dP, noisyA, A, file):
    printParam(dP)
    import keras
    #input = keras.Input(shape=(A.shape[1],),sparse=True)
    input = keras.Input(shape=(A.shape[1],))
    ############
    # Encoder
    ############
    encoded = keras.layers.Dense(A.shape[1]-1, activation=dP.innerActivation,activity_regularizer=keras.regularizers.l1(dP.regL1))(input)
    if dP.deepAutoencoder:
        if dP.linear_net:
            if A.shape[1] > dP.encoded_dim+2:
                for i in range(A.shape[1]-1,dP.encoded_dim+1,-1):
                    encoded = keras.layers.Dense(i-1,  activation=dP.innerActivation,activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
                    encoded = keras.layers.Dropout(dP.dropout)(encoded)
        else:
            for i in range(len(dP.net_arch)):
                encoded = keras.layers.Dense(dP.net_arch[i],  activation=dP.innerActivation,activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
                encoded = keras.layers.Dropout(dP.dropout)(encoded)
        encoded = keras.layers.Dense(dP.encoded_dim,activation=dP.innerActivation,activity_regularizer=keras.regularizers.l1(dP.regL1))(encoded)
        encoded = keras.layers.Dropout(dP.dropout)(encoded)
        
    ############
    # Decoder
    ############
    if dP.deepAutoencoder:
        decoded = keras.layers.Dense(dP.encoded_dim+1,  activation=dP.innerActivation)(encoded)
        if dP.linear_net:
            if A.shape[1] > dP.encoded_dim+2:
                for i in range(dP.encoded_dim+2,A.shape[1],1):
                    decoded = keras.layers.Dense(i, activation=dP.innerActivation)(decoded)
        else:
            for i in range(len(dP.net_arch)-1,-1,-1):
                decoded = keras.layers.Dense(dP.net_arch[i], activation=dP.innerActivation)(decoded)
        decoded = keras.layers.Dense(A.shape[1], activation=dP.activation)(decoded)
    else:
        decoded = keras.layers.Dense(A.shape[1], activation=dP.activation)(encoded)
    
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
    optim = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9,
                beta_2=0.999, epsilon=1e-08,
                amsgrad=False)
    #optim = keras.optimizers.Adam()
    
    if dP.reinforce and os.path.exists(dP.modelName):
        print("  Loading existing DAE model:",dP.modelName,"\n")
        autoencoder = keras.saving.load_model(dP.modelName)
    else:
        print("  Initializing new DAE model:",dP.modelName,"\n")
        autoencoder = keras.Model(input, decoded)
        autoencoder.compile(loss = dP.lossMetric, optimizer = optim)
        
    # TensorBoard callback disabled to reduce per-run overhead
    # (histogram + graph writing on every training run). Re-enable
    # by uncommenting the two lines below and removing "tbLogs = []".
    #tbLog = keras.callbacks.TensorBoard(log_dir=dP.tb_directory, histogram_freq=120,
    #        write_graph=True, write_images=False)
    #tbLogs = [tbLog]
    tbLogs = []
    
    if dP.stopAtBest == True:
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
        tbLogs.append(es)
    if dP.saveBestModel == True:
        mc = keras.callbacks.ModelCheckpoint(dP.modelName, monitor=dP.metricBestModel, mode='min', verbose=1, save_best_only=True)
        tbLogs.append(mc)
    
    log = autoencoder.fit(noisyA, A,
            batch_size=dP.batch_size,
            epochs=dP.epochs,
            callbacks = tbLogs,
            shuffle = True,
            verbose=1,
            validation_split=dP.validation_split)
        
    if dP.saveBestModel:
        val_loss = np.min(np.asarray(log.history['val_loss']))
        tag = "best"
    else:
        val_loss = np.asarray(log.history['val_loss'])[-1]
        tag = "last"
        
    print(f"\n  Autoencoder with {tag} model (val_loss: {val_loss:.4f}) saved in: {dP.modelName}\n")
    autoencoder.save(dP.modelName)
    
    return autoencoder, val_loss

#************************************
# Generate data from Autoencoder
#************************************
def removeSpurious(A_physical, T, dP):
    # Run purely on physical matrix to avoid norm/denorm floating point truncation
    A_min = getAmin(A_physical)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i,j] < A_min[j]:
                T[i,j] = 0
    return T

#**************************************************************************
# Snaps generated data to discrete steps if they are close.
#    PURGES the entire row if the generated data falls too far
#    into the invalid 'void' between steps.
#**************************************************************************
def snap_discrete_features(real_features, synthetic_features, discrete_threshold=10, tolerance=0.15):
    corrected_synthetic = np.copy(synthetic_features)
    num_features = real_features.shape[1]
    valid_mask = np.ones(synthetic_features.shape[0], dtype=bool)

    # --- Identify discrete columns ---
    discrete_cols = []
    # Start at 1 to strictly protect the label column (Col 0)
    for i in range(1, num_features):
        unique_vals = np.unique(real_features[:, i])
        if len(unique_vals) <= discrete_threshold:
            discrete_cols.append(i)
            synth_col = synthetic_features[:, i]
            distances = np.abs(synth_col[:, np.newaxis] - unique_vals)
            min_distances = distances.min(axis=1)
            closest_indices = distances.argmin(axis=1)
            # Scale the tolerance to this column's discrete step, so the
            # purge criterion is relative rather than absolute. A fixed
            # absolute tolerance (e.g. 0.15) is far too tight for a
            # large-magnitude column (values 800-900, step 25) while being
            # reasonable for a small one (values 1-4, step 1). Here
            # `tolerance` is interpreted as a fraction of the smallest gap
            # between adjacent real steps.
            if len(unique_vals) > 1:
                col_step = np.min(np.diff(unique_vals))
            else:
                col_step = 1.0
            col_tol = tolerance * col_step
            valid_mask = valid_mask & (min_distances <= col_tol)
            corrected_synthetic[:, i] = unique_vals[closest_indices]
        else:
            print(f"   [v] Col {i}: CONTINUOUS. Preserving variance.")
    
    # --- Constrain label range per discrete combination ---
    if discrete_cols:
        # For each synthetic row, find the matching discrete combination
        # in real data and constrain col 0 to the observed range
        real_discrete = real_features[:, discrete_cols]
        synth_discrete = corrected_synthetic[:, discrete_cols]
        
        label_col = 0  # The predicted parameter
        # Calculate the global label range to ensure the margin scales correctly
        global_label_range = real_features[:, label_col].max() - real_features[:, label_col].min()

        for row_idx in range(corrected_synthetic.shape[0]):
            if not valid_mask[row_idx]:
                continue
            # Find real rows with the same discrete combination
            match_mask = np.all(real_discrete == synth_discrete[row_idx], axis=1)
            if np.any(match_mask):
                real_labels = real_features[match_mask, label_col]
                label_min = real_labels.min()
                label_max = real_labels.max()
                label_range = label_max - label_min
                # Allow some margin (e.g., 20% local, or 5% global if point is isolated)
                margin = 0.2 * label_range if label_range > 0 else 0.05 * global_label_range
                
                if (corrected_synthetic[row_idx, label_col] < label_min - margin or
                    corrected_synthetic[row_idx, label_col] > label_max + margin):
                    valid_mask[row_idx] = False
    
    purged_synthetic = corrected_synthetic[valid_mask]
    rows_removed = synthetic_features.shape[0] - purged_synthetic.shape[0]
    print(f"   [x] Purged {rows_removed} unphysical rows.")
    
    return purged_synthetic

def generateData(dP, autoencoder, A, M, norm):
    import keras
    num_random = int(0.50 * A.shape[0])
    
    # Clone real rows to keep Labels and Discrete columns correlated
    random_indices = np.random.choice(A.shape[0], size=num_random, replace=True)
    random_seeds = np.copy(A[random_indices])
    
    # Jitter only the continuous columns
    for i in range(1, A.shape[1]):
        unique_vals = np.unique(A[:, i])
        if len(unique_vals) > dP.discreteThreshold:
            variance = dP.percNoiseDistrMax * np.std(A[:, i])
            random_seeds[:, i] += np.random.uniform(-variance, variance, size=num_random)
            
    seeds = np.vstack((A, random_seeds))
    
    #----------------
    # Direct call instead of .predict() to avoid the large per-call
    # overhead of predict() on small batches.
    normDae = keras.ops.convert_to_numpy(autoencoder(seeds.astype(np.float32), training=False))

    # Original call, suitable for very large inputs but with higher overhead
    # Uses batches, rather than full input.
    # Use this for large datasets
    #normDae = autoencoder.predict(seeds)
    #-----------------

    # ---------------------------------------------------------------
    # Discrete columns cannot be reliably reproduced by the autoencoder:
    # the reconstruction drifts off the real steps (badly so for
    # large-magnitude columns), and the drift is then purged downstream
    # by snap_discrete_features. Since the seeds already carry the exact,
    # correlated discrete values (they are clones of real rows and are
    # never jittered above), overwrite the reconstruction with the seed
    # values so discrete features land exactly on real steps.
    # Done here in normalized space, before inverse-transform, to stay
    # consistent with the rest of normDae.
    # ---------------------------------------------------------------
    for i in range(1, A.shape[1]):
        unique_vals = np.unique(A[:, i])
        if len(unique_vals) <= dP.discreteThreshold:
            normDae[:, i] = seeds[:, i]

    if dP.postGenerationNoise:
        noise = np.random.normal(loc=0.0, scale=dP.postGenerationNoiseMax, size=normDae.shape)
        # Protect Label and Discrete features from post-generation noise
        noise[:, 0] = 0.0
        for i in range(1, A.shape[1]):
            unique_vals = np.unique(A[:, i])
            if len(unique_vals) <= dP.discreteThreshold:
                noise[:, i] = 0.0
                
        normDae = normDae + noise
    
    if dP.normalize:
        normDae = norm.transform_inverse(normDae)
    
    return normDae
 
#************************************
# Open Learning Data
#************************************
def readLearnFileDAE(learnFile, newNorm, dP):
    M = readFile(learnFile)
    empty = False

    ind = np.any(M == 0, axis=1)
    ind[0] = False
    M_no_zero_features_mask = ~ind

    ind_labels = (M[:, 0] != 0)
    ind_labels[0] = True
    M_no_zero_labels_mask = ind_labels

    if M_no_zero_features_mask.sum() == 1:
        print("  Matrix with no zeros is empty\n")

    if M_no_zero_labels_mask.sum() == 1:
        print("  Labels in the matrix are all zero.\n")
        empty = True

    if dP.normalize:
        print("  Normalization of feature matrix to 1")
        if newNorm:
            print("  Normalization parameters saved in:", dP.norm_file, "\n")
            norm = Normalizer(M, dP)
            norm.save()
        else:
            print("  Normalization parameters from:", dP.norm_file, "\n")
            with open(dP.norm_file, "rb") as f:
                norm = pickle.load(f)
        M = norm.transform(M)

    if dP.excludeZeroLabels:
        print("  Removing data with zero label.\n")
        M = M[M_no_zero_labels_mask]
    elif dP.excludeZeroFeatures:
        print("  Removing data with zero features.\n")
        M = M[M_no_zero_features_mask]
    elif dP.excludeZeroLabels and dP.excludeZeroFeatures:
        M = M[M_no_zero_labels_mask & M_no_zero_features_mask]

    En = M[0, :]
    A = M[1:, :]
    Cl = M[1:, 0]

    return En, A, M, empty

#************************************
# Plot augmented training data
#************************************
def plotData(dP, A, newA, feat, normFlag, title, plotFile):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import r2_score
    import warnings
    
    pdf = PdfPages(plotFile)
        
    if dP.normalize and normFlag:
        with open(dP.norm_file, "rb") as f:
            norm = pickle.load(f)
        A = norm.transform_inverse(A)
        if newA is not None:
            newA = norm.transform_inverse(newA)
        
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
        
        if newA is not None:
            plt.plot(xA,yA, 'bo', markersize=3)
        
        unique_x_count = len(np.unique(x))
        safe_degree = min(dP.fitPolyDegree, unique_x_count - 1)
        safe_degree = max(0, safe_degree) # Fallback to 0 (flat line) if only 1 unique point exists
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            poly = polyfit.fit(x, y, safe_degree)
        
        plt.plot(np.unique(x), poly(np.unique(x)))
        plt.plot(x,y, 'ro', markersize=5)
        plt.title(title+" - $R^2={0:.3f}$".format(r2_score(y, poly(x))))
        pdf.savefig()
        plt.close()
    pdf.close()
    
#************************************
# Print NN Info
#************************************
def printParam(dP):
    print('  ================================================')
    print('  \033[1m DAE \033[0m - Parameters')
    print('  ================================================')
    print('  Linear Architecture:',dP.linear_net)
    print('  Architecture:', dP.net_arch, '->', dP.encoded_dim)
    print('  Outer Activation function:', dP.activation)
    print('  Inner Activation function:', dP.innerActivation)
    print('  Loss Metric:', dP.lossMetric)
    
    print('  Reg L1:',dP.regL1)
    print('  Dropout:', dP.dropout)
    print('  Learning rate:', dP.l_rate)
    print('  Learning decay rate:', dP.l_rdecay)

    print('  Batch size:', dP.batch_size)
    print('  Epochs:',dP.epochs)
    print('  Min loss dae:',dP.min_loss_dae)
    print('  Num Additions:',dP.numAdditions)
    print('  Num Added Noisy Data Blocks:', dP.numAddedNoisyDataBlocks)
    print('  Max Perc Noise Distribution:', dP.percNoiseDistrMax)
    print('  Post-Generation Noise:',dP.postGenerationNoise)
    print('  Post-Generation Noise Max Perc:',dP.postGenerationNoiseMax)
    
    print('  Type Noise:',dP.typeNoise)
    print('  Num Col Swaps:',dP.numColSwaps)
    
    print('  Remove Spurious:',dP.removeSpurious)
    print('  Normalize:',dP.normalize)
    print('  Normalize Label:',dP.normalizeLabel)

    print('  Metric for Best Model:', dP.metricBestModel)
    
    print('  ================================================\n')

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

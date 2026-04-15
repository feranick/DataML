#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* DataML_KDE
* Generative AI via Kernel Density Estimation
* version: 2026.04.15.2
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, pickle, configparser, getopt
from numpy.polynomial.polynomial import Polynomial as polyfit
from sklearn.neighbors import KernelDensity

from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def DataML_KDE():
    main()

#************************************
# Parameters definition
#************************************
class Conf():
    def __init__(self):
        self.appName = "DataML_KDE"
        confFileName = "DataML_KDE.ini"
        self.configFile = os.path.join(os.getcwd(),confFileName)
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        
        self.model_directory = "./"
        self.modelName = self.model_directory + "model_KDE.pkl"
        self.norm_file = self.model_directory + "norm_file_KDE.pkl"
        self.numLabels = 1

        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
            
    def defKDEPar(self):
        self.conf['Parameters'] = {
            'saveAsTxt' : True,
            'shuffle' : True,
            'kde_bandwidth' : 0.05,
            'kde_kernel' : 'gaussian',
            'numAddedDataBlocks' : 50, 
            'fitPolyDegree' : 3,
            'excludeZeroFeatures' : False,
            'excludeZeroLabels' : True,
            'removeSpurious' : True,
            'normalize' : True,
            'normalizeLabel' : True,
            'discreteThreshold' : 5,
            'plotAugmData' : True
            }
            
    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.saveAsTxt = self.conf.getboolean('Parameters','saveAsTxt')
            self.shuffle = self.conf.getboolean('Parameters','shuffle')
            self.kde_bandwidth = self.conf.getfloat('Parameters','kde_bandwidth')
            self.kde_kernel = self.conf.get('Parameters','kde_kernel')
            self.numAddedDataBlocks = self.conf.getint('Parameters','numAddedDataBlocks')
            self.fitPolyDegree = self.conf.getint('Parameters','fitPolyDegree')
            
            self.excludeZeroFeatures = self.conf.getboolean('Parameters','excludeZeroFeatures')
            self.excludeZeroLabels = self.conf.getboolean('Parameters','excludeZeroLabels')
            self.removeSpurious = self.conf.getboolean('Parameters','removeSpurious')
            self.normalize = self.conf.getboolean('Parameters','normalize')
            self.normalizeLabel = self.conf.getboolean('Parameters','normalizeLabel')
            self.discreteThreshold = self.conf.getint('Parameters','discreteThreshold', fallback=5)
            self.plotAugmData = self.conf.getboolean('Parameters','plotAugmData')
        
        except Exception as e:
            print(" Error in reading configuration file:")
            print(f"  {e}\n")

    def createConfig(self):
        try:
            self.defKDEPar()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
        except Exception as e:
            print("Error in creating configuration file:")
            print(f"  {e}\n")
    
#************************************
# Main
#************************************
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "tag:", ["train", "augment", "generate"])
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
            #try:
            augment(sys.argv[2], True)
            #except Exception as e:
            #   print(f" An error occurred: {e}\n")
            #   sys.exit(2)
        
        if o in ("-g" , "--generate"):
            try:
                generate(sys.argv[2])
            except Exception as e:
                print(f" An error occurred: {e}\n")
                sys.exit(2)

#***********************************************
# Train KDE via final learning file
#***********************************************
def train(learnFile):
    augment(learnFile, False)

#***********************************************
# Generate new sample based on prompt
#***********************************************
def generate(csvFile):
    import pandas as pd
    from datetime import datetime
    dP = Conf()
    
    print(f"  Opening file with prompt samples: {csvFile}")
    dataDf = pd.read_csv(csvFile)
    num_samples_to_generate = dataDf.shape[0]
        
    print("  Loading existing KDE model:",dP.modelName,"\n")
    try:
        with open(dP.modelName, "rb") as f:
            kde = pickle.load(f)
    except:
        print("\033[1m KDE model file not found. Train it first. \033[0m")
        sys.exit()
    
    if dP.normalize:
        try:
            with open(dP.norm_file, "rb") as f:
                norm = pickle.load(f)
        except:
            print("\033[1m pkl file not found \033[0m")
            sys.exit()
    else:
        norm = None
    
    # KDE samples pure distributions rather than predicting Y from X
    print(f"  Generating {num_samples_to_generate} synthetic samples directly from KDE distribution...")
    generated_data = kde.sample(num_samples_to_generate)
    
    if dP.normalize:
        generated_data = norm.transform_inverse(generated_data)
        
    # Populate the output DF
    newDataDf = pd.DataFrame(generated_data, columns=dataDf.columns)
    
    summaryCSVFileRoot = os.path.splitext(csvFile)[0]
    summaryCSVFile = summaryCSVFileRoot+"_KDE_output"+str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))
    
    newDataDf.to_csv(summaryCSVFile, index=False, sep=',')
    print(f" KDE generated samples saved in: {summaryCSVFile}\n")

#***********************************************
# Augment learning data via KDE
#***********************************************
def augment(learnFile, augFlag):
    dP = Conf()
    
    M_raw = readFile(learnFile)
    En_orig = M_raw[0, :]
    
    try:
        En, A, M, empty = readLearnFileKDE(learnFile, True, dP)
        if empty:
            return 1
    except Exception as e:
        print(f" An error occurred during augmentation: {e}\n")
        return 1
    
    rootFile = dP.model_directory + os.path.splitext(os.path.basename(learnFile))[0] + \
            '_numAddedKDE' + str(dP.numAddedDataBlocks * A.shape[0])
    
    if dP.normalize:
        with open(dP.norm_file, "rb") as f:
            norm = pickle.load(f)
        orig_physical_A = norm.transform_inverse(A)
    else:
        orig_physical_A = A
        norm = None
        
    printParam(dP)
    
    # Fit the Kernel Density Estimator
    print("  Fitting Kernel Density Estimator to dataset...")
    kde = KernelDensity(bandwidth=dP.kde_bandwidth, kernel=dP.kde_kernel)
    kde.fit(A)
    
    # Save the fitted KDE model
    with open(dP.modelName, 'wb') as f:
        pickle.dump(kde, f)
    print(f"  KDE model saved to {dP.modelName}")
    
    if not augFlag:
        return
        
    # Generate Synthetic Samples
    n_samples = int(dP.numAddedDataBlocks * A.shape[0])
    print(f"\n  Sampling {n_samples} points from KDE landscape...")
    synthetic_A = kde.sample(n_samples)
    
    # Snap discrete variables back into physical bounds
    print("  Applying physical constraints and boundaries...")
    purged_synthetic_A = snap_discrete_features(orig_physical_A, synthetic_A, dP.discreteThreshold, norm=norm)
    
    # De-normalize generated data for the final file
    if dP.normalize:
        purged_physical_A = norm.transform_inverse(purged_synthetic_A)
    else:
        purged_physical_A = purged_synthetic_A
        
    if dP.removeSpurious:
        purged_physical_A = removeSpurious(orig_physical_A, purged_physical_A)
        print("  Spurious data removed.")

    # Combine original and new data
    newA = np.vstack([orig_physical_A, purged_physical_A])
    newTrain = np.vstack([En_orig, newA])
    
    total_added_rows = purged_physical_A.shape[0]
    print(f"\n  Successfully added {total_added_rows} physically valid data points.")
    
    newFile = rootFile + '_kde_aug' 
    saveLearnFile(dP, newTrain, newFile, "")
    
    if dP.plotAugmData:
        plotData(dP, orig_physical_A, purged_physical_A, True, "KDE Augmented Data", newFile+"_plots.pdf")

#******************************************************
# Utility Functions
#******************************************************
def getAmin(A):
    A_min = []
    for i in range(A.shape[1]):
        non_zero = A[:, i][A[:, i] != 0]
        A_min_single = non_zero.min() if non_zero.size > 0 else 0.0
        A_min.append(A_min_single)
    return np.array(A_min)

def removeSpurious(A_physical, T):
    A_min = getAmin(A_physical)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i,j] < A_min[j]:
                T[i,j] = 0
    return T

def snap_discrete_features(real_features, synthetic_features, discrete_threshold=10, tolerance=0.15, norm=None):
    if norm is not None:
        synthetic_physical = norm.transform_inverse(synthetic_features)
    else:
        synthetic_physical = synthetic_features
        
    corrected_synthetic = np.copy(synthetic_physical)
    num_features = real_features.shape[1]
    valid_mask = np.ones(synthetic_physical.shape[0], dtype=bool)
    
    discrete_cols = []
    for i in range(1, num_features):
        unique_vals = np.unique(real_features[:, i])
        if len(unique_vals) <= discrete_threshold:
            discrete_cols.append(i)
            synth_col = synthetic_physical[:, i]
            distances = np.abs(synth_col[:, np.newaxis] - unique_vals)
            min_distances = distances.min(axis=1)
            closest_indices = distances.argmin(axis=1)
            valid_mask = valid_mask & (min_distances <= tolerance)
            corrected_synthetic[:, i] = unique_vals[closest_indices]
        else:
            pass # Continuous
            
    if discrete_cols:
        real_discrete = real_features[:, discrete_cols]
        synth_discrete = corrected_synthetic[:, discrete_cols]
        label_col = 0  
        global_label_range = real_features[:, label_col].max() - real_features[:, label_col].min()

        for row_idx in range(corrected_synthetic.shape[0]):
            if not valid_mask[row_idx]: continue
            match_mask = np.all(real_discrete == synth_discrete[row_idx], axis=1)
            if np.any(match_mask):
                real_labels = real_features[match_mask, label_col]
                label_min = real_labels.min()
                label_max = real_labels.max()
                label_range = label_max - label_min
                margin = 0.2 * label_range if label_range > 0 else 0.05 * global_label_range
                
                if (corrected_synthetic[row_idx, label_col] < label_min - margin or
                    corrected_synthetic[row_idx, label_col] > label_max + margin):
                    valid_mask[row_idx] = False
    
    purged_synthetic = corrected_synthetic[valid_mask]
    
    # Re-normalize if necessary so the return matches the input space type
    if norm is not None:
        purged_synthetic = norm.transform(purged_synthetic)
        
    rows_removed = synthetic_features.shape[0] - purged_synthetic.shape[0]
    print(f"   [x] Purged {rows_removed} unphysical rows out of bounds.")
    return purged_synthetic

def readLearnFileKDE(learnFile, newNorm, dP):
    M = readFile(learnFile)
    empty = False
    
    if dP.normalize:
        print("  Normalization of feature matrix to 1")
        if newNorm:
            norm = Normalizer(M, dP)
            norm.save()
        else:
            with open(dP.norm_file, "rb") as f:
                norm = pickle.load(f)
        M = norm.transform(M)
    
    ind = np.any(M == 0, axis=1)
    ind[0] = False
    M_no_zero_features = M[~ind]
    
    ind_labels = (M[:, 0] != 0)
    ind_labels[0] = True
    M_no_zero_labels = M[ind_labels]
    
    if dP.excludeZeroLabels: M = M_no_zero_labels
    if dP.excludeZeroFeatures: M = M_no_zero_features
    
    En = M[0,:]
    A = M[1:,:]
    return En, A, M, empty

def plotData(dP, A_physical, newA_physical, feat, title, plotFile):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import r2_score
    
    pdf = PdfPages(plotFile)
        
    for i in range(1, A_physical.shape[1]):
        if feat:
            x = A_physical[:,i]
            y = A_physical[:,0]
            xA = newA_physical[:,i]
            yA = newA_physical[:,0]
            plt.xlabel("col "+str(i)+" - feature parameter")
            plt.ylabel("col 0 - predicted parameter")
        else:
            y = A_physical[:,i]
            x = A_physical[:,0]
            yA = newA_physical[:,i]
            xA = newA_physical[:,0]
            plt.xlabel("col 0 - predicted parameter")
            plt.ylabel("col "+str(i)+" - feature parameter")
        
        plt.plot(xA,yA, 'bo', markersize=3, alpha=0.5, label='KDE Generated')
        
        poly = polyfit.fit(x, y, dP.fitPolyDegree)
        plt.plot(np.unique(x), poly(np.unique(x)), 'g-', label='Polynomial Fit')
        plt.plot(x,y, 'ro', markersize=5, label='Original Data')
        
        plt.title(title+" - $R^2={0:.3f}$".format(r2_score(y, poly(x))))
        pdf.savefig()
        plt.close()
    pdf.close()

def printParam(dP):
    print('  ================================================')
    print('  \033[1m KDE \033[0m - Parameters')
    print('  ================================================')
    print('  Bandwidth (Smoothing):', dP.kde_bandwidth)
    print('  Kernel Type:', dP.kde_kernel)
    print('  Added Data Multiplier:', dP.numAddedDataBlocks)
    print('  Remove Spurious:', dP.removeSpurious)
    print('  Normalize:', dP.normalize)
    print('  Discrete Threshold:', dP.discreteThreshold)
    print('  ================================================\n')

def usage():
    print('\n Usage:\n')
    print(' Fit KDE:')
    print('  DataML_KDE -t <learningFile>\n')
    print(' Augment data from <learningFile> using KDE:')
    print('  DataML_KDE -a <learningFile>\n')
    print(' Generate new raw KDE samples (provide dummy csv for shape/count)')
    print('  DataML_KDE -g <csvlist>\n')

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***************************************************
* DataML Decision Trees - Classifier and Regressor
* v2024.10.16.4
* Uses: sklearn
* By: Nicola Ferralis <feranick@hotmail.com>
***************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, getopt, time, configparser
import platform, pickle, h5py, csv, glob, math
from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def DataML_DT():
    main()

#************************************
# Parameters
#************************************
class Conf():
    def __init__(self):
    
        #############################
        ### Types of estimators:
        ### - RandomForest
        ### - HistGradientBoosting
        ### - GradientBoosting
        ### - DecisionTree
        #############################
        
        self.appName = "DataML_DT"
        confFileName = "DataML_DT.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "./"
        if self.regressor:
            self.mode = "Regressor"
            self.metric = "MAE"
        else:
            self.mode = "Classifier"
            self.metric = "Accuracy"
        
        self.modelNameRoot = "model_DT_"
        self.modelName = self.modelNameRoot + self.typeDT + self.mode + ".pkl"
        self.summaryFileName = self.modelNameRoot + self.typeDT + self.mode + ".csv"
        
        self.tb_directory = "model_DT"
        self.model_name = self.model_directory+self.modelNameRoot
        
        self.model_le = self.model_directory+"model_le.pkl"
        self.model_scaling = self.model_directory+"model_scaling.pkl"
        self.model_pca = self.model_directory+"model_encoder.pkl"
        self.norm_file = self.model_directory+"norm_file.pkl"
                    
        self.rescaleForPCA = False
            
    def datamlDef(self):
        self.conf['Parameters'] = {
            'typeDT' : 'GradientBoosting',
            'regressor' : False,
            'n_estimators' : 4,
            'max_depth' : 7,
            'max_features' : 0.5,
            'epochs' : 100,
            'l_rate' : 0.1,
            'cv_split' : 0.05,
            'trainFullData' : True,
            'fullSizeBatch' : False,
            'batch_size' : 8,
            'numLabels' : 1,
            'normalize' : False,
            'runDimRedFlag' : False,
            'typeDimRed' : 'SparsePCA',
            'numDimRedComp' : 3,
            'plotFeatImportance' : False,
            }
    def sysDef(self):
        self.conf['System'] = {
            'kerasVersion' : 3,
            'n_jobs' : 1
            }

    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.datamlDef = self.conf['Parameters']
            self.sysDef = self.conf['System']
        
            self.typeDT = self.conf.get('Parameters','typeDT')
            self.regressor = self.conf.getboolean('Parameters','regressor')
            self.n_estimators = self.conf.getint('Parameters','n_estimators')
            self.max_depth = self.conf.getint('Parameters','max_depth')
            self.max_features = self.conf.getfloat('Parameters','max_features')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.epochs = self.conf.getint('Parameters','epochs')
            self.cv_split = self.conf.getfloat('Parameters','cv_split')
            self.trainFullData = self.conf.getboolean('Parameters','trainFullData')
            self.fullSizeBatch = self.conf.getboolean('Parameters','fullSizeBatch')
            self.batch_size = self.conf.getint('Parameters','batch_size')
            self.numLabels = self.conf.getint('Parameters','numLabels')
            self.normalize = self.conf.getboolean('Parameters','normalize')
            self.runDimRedFlag = self.conf.getboolean('Parameters','runDimRedFlag')
            self.typeDimRed = self.conf.get('Parameters','typeDimRed')
            self.numDimRedComp = self.conf.getint('Parameters','numDimRedComp')
            self.plotFeatImportance = self.conf.getboolean('Parameters','plotFeatImportance')
            
            self.kerasVersion = self.conf.getint('System','kerasVersion')
            self.n_jobs = self.conf.getint('System','n_jobs')
            
        except:
            print(" Error in reading configuration file. Please check it\n")

    # Create configuration file
    def createConfig(self):
        try:
            self.datamlDef()
            self.sysDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
        except:
            print("Error in creating configuration file")

#************************************
# Main
#************************************
def main():
    dP = Conf()
    start_time = time.perf_counter()
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],
            "tpbvcah:", ["train", "predict", "batch", "validbatch", "comp", "autoencoder", "help"])
    except:
        usage(dP.appName)
        sys.exit(2)

    if opts == []:
        usage(dP.appName)
        sys.exit(2)

    for o, a in opts:
        if o in ("-t" , "--train"):
            #try:
            if len(sys.argv)<4:
                train(sys.argv[2], None, None)
            else:
                if len(sys.argv)<5:
                    train(sys.argv[2], sys.argv[3], None)
                else:
                    train(sys.argv[2], sys.argv[3], sys.argv[4])
            #except:
            #    usage(dP.appName)
            #    sys.exit(2)

        if o in ("-p" , "--predict"):
            #try:
            if len(sys.argv)<4:
                predict(sys.argv[2], None)
            else:
                predict(sys.argv[2], sys.argv[3])
            #except:
            #    usage(dP.appName)
            #    sys.exit(2)

        if o in ("-b" , "--batch"):
            #try:
            if len(sys.argv)<4:
                batchPredict(sys.argv[2], None)
            else:
                batchPredict(sys.argv[2], sys.argv[3])
            #except:
            #    usage(dP.appName)
            #    sys.exit(2)
            
        if o in ("-v" , "--validbatch"):
            #try:
            if len(sys.argv)<4:
                validBatchPredict(sys.argv[2], None)
            else:
                validBatchPredict(sys.argv[2], sys.argv[3])
            #except:
            #    usage(dP.appName)
            #    sys.exit(2)
                
        if o in ["-c" , "--comp"]:
            try:
                if len(sys.argv)<4:
                    prePCA(sys.argv[2], None, dP)
                else:
                    prePCA(sys.argv[2], sys.argv[3], dP)
            except:
                usage(dP.appName)
                sys.exit(2)
            
        if o in ["-a" , "--autoencoder"]:
            try:
                if len(sys.argv)<4:
                    preAutoencoder(sys.argv[2], None, dP)
                else:
                    preAutoencoder(sys.argv[2], sys.argv[3], dP)
            except:
                usage(dP.appName)
                sys.exit(2)

    total_time = time.perf_counter() - start_time
    print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

#************************************
# Training
#************************************
def train(learnFile, testFile, normFile):
    dP = Conf()
   
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from statistics import mean, stdev
    from sklearn.metrics import accuracy_score, mean_absolute_error
    
    learnFileRoot = os.path.splitext(learnFile)[0]

    En, A, Cl = readLearnFile(learnFile, dP)
    if testFile is not None:
        En_test, A_test, Cl_test = readLearnFile(testFile, dP)
        totA = np.vstack((A, A_test))
        totCl = np.append(Cl, Cl_test)
    else:
        totA = A
        totCl = Cl
        A, Cl, A_test, Cl_test, _ = formatSubset(A, Cl, dP.cv_split)
        
    if dP.trainFullData:
        A = totA
        Cl = totCl
        
    print("  Data size:", A.shape)
    print("  Number of learning labels: {0:d}".format(int(dP.numLabels)))
    print("  Total number of points per data:",En.size)
    
    if testFile is not None or dP.trainFullData:
        print("\n  Testing set file:",testFile)
        print("  Training set file datapoints:", A.shape[0])
        print("  Testing set file datapoints:", A_test.shape[0])
    else:
        print("\n  Testing data set from training file:",dP.cv_split*100,"%")
        print("  Training set file datapoints:", math.floor(A.shape[0]*(1-dP.cv_split)))
        print("  Testing set datapoints:", math.ceil(A.shape[0]*dP.cv_split),"\n")
    
    if dP.regressor:
        Cl2 = np.copy(Cl)
        #if testFile is not None:
        Cl2_test = np.copy(Cl_test)
    else:
        #************************************
        # Label Encoding
        #************************************
        '''
        # sklearn preprocessing is only for single labels
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        totCl2 = le.fit_transform(totCl)
        Cl2 = le.transform(Cl)
        if testFile is not None:
            Cl2_test = le.transform(Cl_test)
        '''
        
        le = MultiClassReductor()
        le.fit(np.unique(totCl, axis=0))
        Cl2 = le.transform(Cl)
        
        print("  Number unique classes (training): ", np.unique(Cl).size)
        
        #if testFile is not None:
        Cl2_test = le.transform(Cl_test)
        print("  Number unique classes (validation):", np.unique(Cl_test).size)
        print("  Number unique classes (total): ", np.unique(totCl).size)
            
        print("\n  Label encoder saved in:", dP.model_le,"\n")
        with open(dP.model_le, 'ab') as f:
            pickle.dump(le, f)

    #************************************
    # Run PCA if needed.
    #************************************
    if dP.runDimRedFlag:
        print("  Dimensionality Reduction via:",dP.typeDimRed,"\n")
        if dP.typeDimRed == 'Autoencoder':
            A = runAutoencoder(A, dP)
        else:
            A = runPCA(A, dP.numDimRedComp, dP)
            #if testFile is not None:
            A_test = runPCAValid(A_test, dP)
    
    #************************************
    # Training
    #************************************
    if dP.fullSizeBatch:
        dP.batch_size = A.shape[0]
        
    if dP.regressor:
        if dP.typeDT == 'RandomForest':
            dt = RandomForestRegressor(max_depth=dP.max_depth, n_estimators = dP.n_estimators, random_state=0, max_features = dP.max_features, verbose=2, n_jobs=dP.n_jobs)
        if dP.typeDT == 'HistGradientBoosting':
            dt = HistGradientBoostingRegressor(max_depth=dP.max_depth, max_iter=dP.epochs, max_features = dP.max_features, verbose = 2, learning_rate=dP.l_rate, l2_regularization=0.0,)
        if dP.typeDT == 'GradientBoosting':
            if dP.max_features == 0:
                dP.max_features = None
            dt = GradientBoostingRegressor(n_estimators = dP.epochs, max_depth=dP.max_depth, max_features = dP.max_features, verbose = 2, learning_rate=dP.l_rate)
        if dP.typeDT == 'DecisionTree':
            if dP.max_features == 0:
                dP.max_features = None
            dt = DecisionTreeRegressor(max_depth=dP.max_depth, max_features = dP.max_features)
    else:
        if dP.typeDT == 'RandomForest':
            dt = RandomForestClassifier(max_depth=dP.max_depth, n_estimators = dP.n_estimators, random_state=0, max_features = dP.max_features, verbose=2, n_jobs=dP.n_jobs, oob_score=False)
        if dP.typeDT == 'HistGradientBoosting':
            dt = HistGradientBoostingClassifier(max_depth=dP.max_depth, max_iter=dP.epochs, max_features = dP.max_features, verbose = 2, learning_rate=dP.l_rate, l2_regularization=0.0)
        if dP.typeDT == 'GradientBoosting':
            dt = GradientBoostingClassifier(n_estimators = dP.epochs, max_depth=dP.max_depth, max_features = dP.max_features, verbose = 2, learning_rate=dP.l_rate)
        if dP.typeDT == 'DecisionTree':
            if dP.max_features == 0:
                dP.max_features = None
            dt = DecisionTreeClassifier(max_depth=dP.max_depth, max_features = dP.max_features)
    
    dt.fit(A, Cl2)
        
    print("\n  ",dP.typeDT+dP.mode,"model saved in:", dP.modelName)
    with open(dP.modelName,'wb') as f:
        pickle.dump(dt, f)

    if dP.regressor:
        pred = dt.predict(A_test)
        score = mean_absolute_error(pred, Cl_test)
    else:
        pred = le.inverse_transform_bulk(dt.predict(A_test))
        pred_classes = le.inverse_transform_bulk(dt.classes_)
        proba = dt.predict_proba(A_test)
        score = accuracy_score(pred, Cl_test)
        
    delta = pred - Cl_test
    
    printParamDT(dP)
    
    print('\n  ================================================================================')
    print('  \033[1m',dP.typeDT,dP.mode,'\033[0m- Results')
    print('  ================================================================================')
    print('   Real class\t| Predicted class\t| Delta')
    print('  --------------------------------------------------------------------------------')
    for i in range(len(pred)):
        print("   {0:.2f}\t| {1:.2f}\t\t| {2:.2f}".format(Cl_test[i], pred[i], delta[i]))
    print('  --------------------------------------------------------------------------------')
    print('  ',dP.metric,'= {0:.4f}'.format(score))
    print('   R^2 = {0:.4f}'.format(dt.score(A_test, Cl2_test)))
    print('   Average Delta: {0:.2f}, StDev = {1:.2f}'.format(mean(delta), stdev(delta)))
    
    if not dP.regressor:
        print('\n  ================================================================================')
        print('   Real class\t| Predicted class\t| Probability')
        print('  --------------------------------------------------------------------------------')
    
        for i in range(len(pred)):
            ind = np.where(proba[i]==np.max(proba[i]))[0]
            for j in range(len(ind)):
                print("   {0:.2f}\t| {1:.2f}\t\t| {2:.2f} ".format(Cl_test[i], pred_classes[ind[j]], 100*proba[i][ind[j]]))
            print("")
    
    print('  ================================================================================\n')
    
    if dP.plotFeatImportance and dP.typeDT == 'RandomForest':
        plotImportances(dt, A_test, Cl2_test, dP)
    
    print('  Scikit-learn v.',str(sklearn.__version__),'\n')
    
#************************************
# Prediction
#************************************
def predict(testFile, normFile):
    dP = Conf()
    import sklearn
    R, _ = readTestFile(testFile)

    if normFile is not None:
        try:
            with open(normFile, "rb") as f:
                norm = pickle.load(f)
            print("  Opening pkl file with normalization data:",normFile)
            print("  Normalizing validation file for prediction...\n")
            R = norm.transform_valid_data(R)
        except:
            print("\033[1m pkl file not found \033[0m")
            return
     
    if dP.runDimRedFlag:
        R = runPCAValid(R, dP)
            
    with open(dP.modelName, "rb") as f:
        dt = pickle.load(f)
    
    if dP.regressor:
        pred = dt.predict(R)
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        pred = le.inverse_transform_bulk(dt.predict(R))
        pred_classes = le.inverse_transform_bulk(dt.classes_)
        proba = dt.predict_proba(R)
        
    print('\n  ================================================================================')
    print('  \033[1m',dP.typeDT,dP.mode,'\033[0m')
    print('  ================================================================================')
    if dP.regressor:
        print('   Filename\t\t| Prediction')
        print('  --------------------------------------------------------------------------------')
        print("   {0:s}\t| {1:.2f}  ".format(testFile, pred[0]))
    else:
        print('   Filename\t\t| Prediction\t| Probability')
        print('  --------------------------------------------------------------------------------')
        ind = np.where(proba[0]==np.max(proba[0]))[0]
        for j in range(len(ind)):
            print("   {0:s}\t| {1:.2f}\t| {2:.2f} ".format(testFile, pred_classes[ind[j]], 100*proba[0][ind[j]]))
        print("")
    print('  ================================================================================\n')
    print('  Scikit-learn v.',str(sklearn.__version__),'\n')

#************************************
# Batch Prediction
#************************************
def batchPredict(folder, normFile):
    dP = Conf()
    import sklearn
    
    if normFile is not None:
        try:
            with open(normFile, "rb") as f:
                norm = pickle.load(f)
            print("  Opening pkl file with normalization data:",normFile)
            print("  Normalizing validation file for prediction...\n")
        except:
            print("\033[1m pkl file not found \033[0m")
            return
            
    with open(dP.modelName, "rb") as f:
        dt = pickle.load(f)
    
    summaryFile = np.array([['Folder:',folder,''],['DataML_DT',dP.typeDT,dP.mode]])
    
    if dP.regressor:
        summaryFile = np.vstack((summaryFile,['File Name','Predicted Value','']))
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        summaryFile = np.vstack((summaryFile,['File Name','Predicted Value','Probability %']))
     
    fileName = []
    pred = []
    proba = []
    for file in glob.glob(folder+'/*.txt'):
        R, good = readTestFile(file)
        if  normFile is not None:
            R = norm.transform_valid_data(R)
            
        if dP.runDimRedFlag:
            R = runPCAValid(R, dP)
        
        if good:
            if dP.regressor:
                pred.append(dt.predict(R))
            else:
                pred.append(le.inverse_transform_bulk(dt.predict(R)))
                pred_classes = le.inverse_transform_bulk(dt.classes_)
                proba.append(dt.predict_proba(R))
            fileName.append(file)
    
    print('\n  ================================================================================')
    print('  \033[1m',dP.typeDT,dP.mode,'\033[0m')
    print('  ================================================================================')
    if dP.regressor:
        print('   Filename\t| Prediction')
        print('  --------------------------------------------------------------------------------')
        for i in range(0,len(pred)):
            print("   {0:s}\t| {1:.2f}  ".format(fileName[i], pred[i][0]))
            summaryFile = np.vstack((summaryFile,[fileName[i],pred[i][0],'']))
    else:
        print('   Filename\t\t| Prediction\t| Probability')
        print('  --------------------------------------------------------------------------------')
        for i in range(0,len(pred)):
            ind = np.where(proba[i][0]==np.max(proba[i][0]))[0]
            for j in range(len(ind)):
                print("   {0:s}\t| {1:.2f}\t| {2:.2f} ".format(fileName[i], pred_classes[ind[j]], 100*proba[i][0][ind[j]]))
                summaryFile = np.vstack((summaryFile,[fileName[i],pred_classes[ind[j]],100*proba[i][0][ind[j]]]))
        print("")
    print('  ================================================================================\n')
    
    saveSummaryFile(summaryFile, dP)
    print('  Scikit-learn v.',str(sklearn.__version__),'\n')

#************************************************************
# Batch Prediction using validation data (with real values)
#************************************************************
def validBatchPredict(testFile, normFile):
    dP = Conf()
    En_test, A_test, Cl_test = readLearnFile(testFile, dP)
    import sklearn
    
    if normFile is not None:
        try:
            with open(normFile, "rb") as f:
                norm = pickle.load(f)
            print("  Opening pkl file with normalization data:",normFile)
            print("  Normalizing validation file for prediction...\n")
        except:
            print("\033[1m pkl file not found \033[0m")
            return
            
    with open(dP.modelName, "rb") as f:
        dt = pickle.load(f)
        
    if dP.runDimRedFlag:
        A_test = runPCAValid(A_test, dP)
    
    summaryFile = np.array([['File:',testFile,''],['DataML_DT',dP.typeDT,dP.mode]])
    
    if dP.regressor:
        pred = dt.predict(A_test)
        summaryFile = np.vstack((summaryFile,['Index','Predicted Value','']))
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        pred = le.inverse_transform_bulk(dt.predict(A_test))
        pred_classes = le.inverse_transform_bulk(dt.classes_)
        proba = dt.predict_proba(A_test)
        summaryFile = np.vstack((summaryFile,['Index','Predicted Value','Probability %']))
    
    print('  ================================================================================')
    print('  \033[1m',dP.typeDT,dP.mode,'\033[0m')
    print('  ================================================================================')
    if dP.regressor:
        print('   Prediction')
        print('  --------------------------------------------------------------------------------')
        for i in range(0,len(pred)):
            print("   {0:.2f}  ".format(pred[i]))
            summaryFile = np.vstack((summaryFile,[pred[i],'','']))
    else:
        print('   Prediction\t| Probability')
        print('  --------------------------------------------------------------------------------')
        for i in range(0,len(pred)):
            ind = np.where(proba[i]==np.max(proba[i]))[0]
            for j in range(len(ind)):
                print("   {0:.2f}\t| {1:.2f} ".format(pred_classes[ind[j]], 100*proba[i][ind[j]]))
                summaryFile = np.vstack((summaryFile,[i, pred_classes[ind[j]],100*proba[i][ind[j]]]))
            print("")
        
    print('  ================================================================================\n')
    
    saveSummaryFile(summaryFile, dP)
    print('  Scikit-learn v.',str(sklearn.__version__),'\n')
    
#***********************************************************
# Save Plots with the model importance
#************************************************************
def plotImportances(dt, A_test, Cl_test, dP):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.inspection import permutation_importance

    feature_names = [f"feature {i}" for i in range(A_test.shape[1])]
    
    importances = dt.feature_importances_
    std = np.std([tree.feature_importances_ for tree in dt.estimators_], axis=0)

    forest_importances1 = pd.Series(importances, index=feature_names)
    
    fig, ax = plt.subplots()
    forest_importances1.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using mean decrease in impurity")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    fig.savefig('model_'+dP.typeDT+dP.mode+'_importances_MDI' + '.png', dpi = 160, format = 'png')  # Save plot
    
    
    result = permutation_importance(
        dt, A_test, Cl_test, n_repeats=100, random_state=42, n_jobs=2)
            
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
                
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    
    fig.savefig('model_'+dP.typeDT+dP.mode+'_importances_Perm' + '.png', dpi = 160, format = 'png')  # Save plot
    
    print("  Feature Importance plots saved\n")


#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

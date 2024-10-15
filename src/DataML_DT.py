#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***************************************************
* DataML Decision Trees - Classifier and Regressor
* v2024.10.15.2
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
        self.appName = "DataML_DT"
        confFileName = "DataML-DT.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "./"
        if self.regressor:
            self.summaryFileName = "summary_DT_regressor.csv"
            self.modelName = "model_DT_regressor.pkl"
        else:
            self.summaryFileName = "summary_DT_classifier.csv"
            self.modelName = "model_DT_classifier.pkl"
        
        self.tb_directory = "model_DT"
        self.model_name = self.model_directory+self.modelName
        
        if self.kerasVersion == 3:
            self.model_name = os.path.splitext(self.model_name)[0]+".keras"
        
        self.model_le = self.model_directory+"model_le.pkl"
        self.model_scaling = self.model_directory+"model_scaling.pkl"
        self.model_pca = self.model_directory+"model_encoder.pkl"
        self.norm_file = self.model_directory+"norm_file.pkl"
                    
        self.rescaleForPCA = False
            
    def datamlDef(self):
        self.conf['Parameters'] = {
            'regressor' : False,
            'trainFullData' : True,
            'epochs' : 200,
            'cv_split' : 0.05,
            'fullSizeBatch' : False,
            'batch_size' : 8,
            'numLabels' : 1,
            'normalize' : False,
            'runDimRedFlag' : False,
            'typeDimRed' : 'SparsePCA',
            'numDimRedComp' : 3,
            }
    def sysDef(self):
        self.conf['System'] = {
            'kerasVersion' : 3,
            'fixTFseed' : True,
            }

    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.datamlDef = self.conf['Parameters']
            self.sysDef = self.conf['System']
        
            self.regressor = self.conf.getboolean('Parameters','regressor')
            self.trainFullData = self.conf.getboolean('Parameters','trainFullData')
            self.epochs = self.conf.getint('Parameters','epochs')
            self.cv_split = self.conf.getfloat('Parameters','cv_split')
            self.fullSizeBatch = self.conf.getboolean('Parameters','fullSizeBatch')
            self.batch_size = self.conf.getint('Parameters','batch_size')
            self.numLabels = self.conf.getint('Parameters','numLabels')
            self.runDimRedFlag = self.conf.getboolean('Parameters','runDimRedFlag')
            self.typeDimRed = self.conf.get('Parameters','typeDimRed')
            self.numDimRedComp = self.conf.getint('Parameters','numDimRedComp')
            self.normalize = self.conf.getboolean('Parameters','normalize')
            
            self.kerasVersion = self.conf.getint('System','kerasVersion')
            self.fixTFseed = self.conf.getboolean('System','fixTFseed')
                        
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
            "tpbvcarh:", ["train", "predict", "batch", "validbatch", "comp", "autoencoder", "rforest", "help"])
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
            try:
                if len(sys.argv)<4:
                    predict(sys.argv[2], None)
                else:
                    predict(sys.argv[2], sys.argv[3])
            except:
                usage(dP.appName)
                sys.exit(2)

        if o in ("-b" , "--batch"):
            try:
                if len(sys.argv)<4:
                    batchPredict(sys.argv[2], None)
                else:
                    batchPredict(sys.argv[2], sys.argv[3])
            except:
                usage(dP.appName)
                sys.exit(2)
            
        if o in ("-v" , "--validbatch"):
            try:
                if len(sys.argv)<4:
                    validBatchPredict(sys.argv[2], None)
                else:
                    validBatchPredict(sys.argv[2], sys.argv[3])
            except:
                usage(dP.appName)
                sys.exit(2)
                
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
            
        if o in ["-r" , "--rforest"]:
            try:
                if len(sys.argv)<4:
                    preDT(sys.argv[2], None, dP)
                else:
                    preDT(sys.argv[2], sys.argv[3], dP)
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
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from statistics import mean, stdev
    
    learnFileRoot = os.path.splitext(learnFile)[0]

    En, A, Cl = readLearnFile(learnFile, dP)
    if testFile is not None:
        En_test, A_test, Cl_test = readLearnFile(testFile, dP)
        totA = np.vstack((A, A_test))
        totCl = np.append(Cl, Cl_test)
    else:
        A_train, Cl_train, A_test, Cl_test, _ = formatSubset(A, Cl, dP.cv_split)
        totA = A
        totCl = Cl
        
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
        
    max_depth = 5
    n_estimators = 10
    n_jobs = 1

    if dP.regressor:
        rf = RandomForestRegressor(max_depth=max_depth, n_estimators = n_estimators, random_state=0, verbose=2, n_jobs=n_jobs)
        tag = "Regressor"
    else:
        rf = RandomForestClassifier(max_depth=max_depth, n_estimators = n_estimators, random_state=0, verbose=2, n_jobs=n_jobs)
        tag = "Classifier"
    
    
    rf.fit(A, Cl2)
        
    print("\n  Random Forest", tag,"model saved in:", dP.modelName)
    with open(dP.modelName,'wb') as f:
        pickle.dump(rf, f)

    pred = le.inverse_transform_bulk(rf.predict(A_test))
    delta = pred - Cl_test
        
    print('\n  ================================================================================')
    print('  \033[1m Random Forest \033[0m -',tag,'Prediction')
    print('  ================================================================================')
    print('   Real class\t| Predicted class\t| Delta')
    print('  --------------------------------------------------------------------------------')
    for i in range(len(pred)):
        print("   {0:.2f}\t| {1:.2f}\t\t| {2:.2f}".format(Cl_test[i], pred[i], delta[i]))
    print('  --------------------------------------------------------------------------------')
    print('   Average Delta: {0:.2f}, StDev = {1:.2f}'.format(mean(delta), stdev(delta)))
    print('   R^2: {0:.4f}'.format(rf.score(A_test, Cl2_test)))
    print('  --------------------------------------------------------------------------------\n')
    print('  Scikit-learn v.',str(sklearn.__version__),'\n')
    
#************************************
# Prediction
#************************************
def predict(testFile, normFile):
    dP = Conf()
    R, _ = readTestFile(testFile, dP)

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
        
    if dP.regressor:
        predictions, _ = getPredictions(R, loadModel(dP), dP)
        #predictions = model.predict(R).flatten()[0]
        print('\n  ==========================================================')
        print('  \033[1m MLP - Regressor\033[0m - Prediction')
        print('  ==========================================================')
        if normFile is not None:
            predValue = norm.transform_inverse_single(predictions.flatten()[0])
            print('\033[1m\n  Predicted value = {0:.2f}\033[0m (normalized: {1:.2f})\n'.format(predValue, predictions))
        else:
            predValue = predictions.flatten()[0]
            print('\033[1m\n  Predicted value (normalized) = {0:.2f}\033[0m\n'.format(predValue))
        print('  ==========================================================\n')
        
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        predictions, _ = getPredictions(R, loadModel(dP), dP)
        pred_class = np.argmax(predictions)
        if dP.useTFlitePred:
            predProb = round(100*predictions[0][pred_class]/255,2)
        else:
            predProb = round(100*predictions[0][pred_class],2)
        rosterPred = np.where(predictions[0]>0.1)[0]

        print('\n  ==========================================================')
        print('  \033[1m MLP - Classifier\033[0m - Prediction')
        print('  ==========================================================')

        if dP.numLabels == 1:
            if pred_class.size >0:
                if normFile is not None:
                    predValue = norm.transform_inverse_single(le.inverse_transform([pred_class])[0])
                    print('\033[1m\n  Predicted value = {0:.2f} (probability = {1:.2f}%)\033[0m\n'.format(predValue, predProb))
                else:
                    predValue = le.inverse_transform([pred_class])[0]
                    print('\033[1m\n  Predicted value (normalized) = {0:.2f} (probability = {1:.2f}%)\033[0m\n'.format(predValue, predProb))
            else:
                predValue = 0
                print('\033[1m\n  No predicted value (probability = {0:.2f}%)\033[0m\n'.format(predProb))
            print('  ==========================================================\n')

        else:
            print('\n ============================================')
            print('\033[1m' + ' Predicted value \033[0m(probability = ' + str(predProb) + '%)')
            print(' ============================================\n')
            print("  1:", str(predValue[0]),"%")
            print("  2:",str(predValue[1]),"%")
            print("  3:",str((predValue[1]/0.5)*(100-99.2-.3)),"%\n")
            print(' ============================================\n')

#************************************
# Batch Prediction
#************************************
def batchPredict(folder, normFile):
    dP = Conf()
    model = loadModel(dP)

    if normFile is not None:
        try:
            with open(normFile, "rb") as f:
                norm = pickle.load(f)
            print("  Opening pkl file with normalization data:",normFile,"\n")
        except:
            print("\033[1m" + " pkl file not found \n" + "\033[0m")
            return
            
    fileName = []
    for file in glob.glob(folder+'/*.txt'):
        R, good = readTestFile(file, dP)
        if  normFile is not None:
            R = norm.transform_valid_data(R)
            
        if dP.runDimRedFlag:
            R = runPCAValid(R, dP)
        
        if good:
            try:
                predictions = np.vstack((predictions,getPredictions(R, model, dP)[0].flatten()))
            except:
                predictions = np.array([getPredictions(R, model, dP)[0].flatten()])
            fileName.append(file)
    
    if dP.regressor:
        summaryFile = np.array([['DataML','Regressor',''],['Filename','Prediction','']])
        print('\n  ================================================================================')
        print('  \033[1m MLP - Regressor\033[0m - Batch Prediction')
        print('  ================================================================================')
        for i in range(0,len(predictions)):
            if normFile is not None:
                predValue = norm.transform_inverse_single(predictions[i][0])
            else:
                predValue = predictions[i][0]
            
            print("  {0:s} | {1:.2f}  ".format(fileName[i], predValue))
            summaryFile = np.vstack((summaryFile,[fileName[i], predValue,'']))
        print('  ================================================================================')
    else:
        summaryFile = np.array([['DataML','Classifier',''],['Real Class','Predicted Class', 'Probability']])
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        #predictions, _ = getPredictions(A_test, model,dP)
        #predictions = model.predict(A_test)
        print('\n  ================================================================================')
        print('  \033[1m MLP - Classifier\033[0m - Batch Prediction')
        print('  ================================================================================')
        print("  Real class\t| Predicted class\t| Probability")
        print("  ---------------------------------------------------")
        for i in range(predictions.shape[0]):
            predClass = np.argmax(predictions[i])
            predProb = round(100*predictions[i][predClass],2)
            if normFile is not None:
                predValue = norm.transform_inverse_single(le.inverse_transform([predClass])[0])
                realValue = norm.transform_inverse_single(Cl_test[i])
            else:
                predValue = le.inverse_transform([predClass])[0]
                realValue = Cl_test[i]
            print("  {0:.2f}\t\t| {1:.2f}\t\t\t| {2:.2f}".format(realValue, predValue, predProb))
            summaryFile = np.vstack((summaryFile,[realValue,predValue,predProb]))
        print('  ========================================================\n')

    import pandas as pd
    df = pd.DataFrame(summaryFile)
    df.to_csv(dP.summaryFileName, index=False, header=False)
    print("\n Prediction summary saved in:",dP.summaryFileName,"\n")

#***********************************************************
# Batch Prediction using validation data (with real values)
#************************************************************
def validBatchPredict(testFile, normFile):
    dP = Conf()
    En_test, A_test, Cl_test = readLearnFile(testFile, dP)
    model = loadModel(dP)

    if normFile is not None:
        try:
            with open(normFile, "rb") as f:
                norm = pickle.load(f)
            print("  Opening pkl file with normalization data:",normFile,"\n")
        except:
            print("\033[1m" + " pkl file not found \n" + "\033[0m")
            return
    
    if dP.runDimRedFlag:
        A_test = runPCAValid(A_test, dP)

    covMatrix = np.empty((0,2))
    
    if dP.regressor:
        summaryFile = np.array([['DataML','Regressor','','',''],['Real Value','Prediction','val_loss','val_abs_mean_error','deviation %']])
        predictions, _ = getPredictions(A_test, model, dP)
        
        score = model.evaluate(A_test, Cl_test, batch_size=dP.batch_size, verbose = 0)
        print('  ==========================================================')
        print('  \033[1m MLP - Regressor\033[0m - Batch Prediction')
        print('  ==========================================================')
        print("  \033[1mOverall val_loss:\033[0m {0:.4f}; \033[1moverall val_abs_mean_loss:\033[0m {1:.4f}\n".format(score[0], score[1]))
        print('  ==========================================================')
        
        print('  ===========================================================================')
        print("  Real value | Predicted value | val_loss | val_mean_abs_err | % deviation ")
        print("  ---------------------------------------------------------------------------")
        for i in range(0,len(predictions)):
            score = model.evaluate(np.array([A_test[i]]), np.array([Cl_test[i]]), batch_size=dP.batch_size, verbose = 0)
            if normFile is not None:
                predValue = norm.transform_inverse_single(predictions[i][0])
                realValue = norm.transform_inverse_single(Cl_test[i])
                print("  {0:.3f} ({1:.3f})  |  {2:.3f} ({3:.3f})  | {4:.4f}  |  {5:.4f} | {6:.2f}".format(realValue,Cl_test[i],predValue,
                        Cl_test[i], norm.transform_inverse_single(predictions[i][0]), predictions[i][0], score[0], score[1],predictions[i][0],
                        score[0],score[1], 100*score[1]/norm.transform_inverse_single(Cl_test[i])))
            else:
                realValue = Cl_test[i]
                predValue = predictions[i][0]
                print("  {0:.3f}\t| {1:.3f}\t| {2:.4f}\t| {3:.4f}\t| {4:.1f}".format(realValue, predValue, score[0], score[1], 100*score[1]/realValue))
                
            summaryFile = np.vstack((summaryFile,[realValue,predValue,score[0], score[1],100*score[1]/realValue]))
            
            covMatrix = np.vstack((covMatrix,[realValue,predValue]))
        print('  ===========================================================================\n')
    else:
        summaryFile = np.array([['DataML','Classifier',''],['Real Class','Predicted Class', 'Probability']])
        
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        predictions, _ = getPredictions(A_test, model,dP)
        #predictions = model.predict(A_test)
        print('  ========================================================')
        print('  \033[1m MLP - Classifier\033[0m - Batch Prediction')
        print('  ========================================================')
        print('  Real class\t| Predicted class\t| Probability')
        print('  ---------------------------------------------------')
        for i in range(predictions.shape[0]):
            predClass = np.argmax(predictions[i])
            predProb = round(100*predictions[i][predClass],2)
            if normFile is not None:
                predValue = norm.transform_inverse_single(le.inverse_transform([predClass])[0])
                realValue = norm.transform_inverse_single(Cl_test[i])
            else:
                predValue = le.inverse_transform([predClass])[0]
                realValue = Cl_test[i]
            print("  {0:.2f}\t\t| {1:.2f}\t\t\t| {2:.2f}".format(realValue, predValue, predProb))
            summaryFile = np.vstack((summaryFile,[realValue,predValue,predProb]))
            covMatrix = np.vstack((covMatrix,[realValue,predValue]))
            
        print('  ========================================================\n')
    
    from scipy.stats import pearsonr, spearmanr
    pearsonr_corr, _ = pearsonr(covMatrix[:,0],covMatrix[:,1])
    summaryFile = np.vstack((summaryFile,['PearsonR',pearsonr_corr,'','','']))
    spearmanr_corr, _ = pearsonr(covMatrix[:,0],covMatrix[:,1])
    summaryFile = np.vstack((summaryFile,['SpearmanR',spearmanr_corr,'','','']))
    print(" PearsonR correlation: {0:0.3f}".format(pearsonr_corr))
    print(" SpearmanR correlation: {0:0.4f}".format(spearmanr_corr))
    
    import pandas as pd
    df = pd.DataFrame(summaryFile)
    df.to_csv(dP.summaryFileName, index=False, header=False)
    print("\n Prediction summary saved in:",dP.summaryFileName,"\n")
    
#********************************************************************************
# Perform Random Forest - Preview
#********************************************************************************
def preDT(learnFile, validFile, dP):
    En, A, Cl = readLearnFile(learnFile, dP)
    if validFile is not None:
        En_test, A_test, Cl_test = readLearnFile(validFile, dP)
    else:
        En_test, A_test, Cl_test = None, None, None
    
    if dP.runDimRedFlag:
        print("  Dimensionality Reduction via:",dP.typeDimRed,"\n")
        if dP.typeDimRed == 'Autoencoder':
            A = runAutoencoder(A, dP)
        else:
            A = runPCA(A, dP.numDimRedComp, dP)
            if validFile is not None:
                A_test = runPCAValid(A_test, dP)
    
    runDT(A, Cl, A_test, Cl_test,dP)

def runDT(A, Cl, A_test, Cl_test, dP):
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from statistics import mean, stdev

    max_depth = 5
    n_estimators = 10
    n_jobs = 1

    if dP.regressor:
        rf = RandomForestRegressor(max_depth=max_depth, n_estimators = n_estimators, random_state=0, verbose=2, n_jobs=n_jobs)
        tag = "Regressor"
    else:
        rf = RandomForestClassifier(max_depth=max_depth, n_estimators = n_estimators, random_state=0, verbose=2, n_jobs=n_jobs)
        tag = "Classifier"
        
    rf.fit(A, Cl)
    print("\n  Random Forest", tag,"model saved in:", dP.modelName)
    with open(dP.modelname,'wb') as f:
        pickle.dump(rf, f)

    if A_test is not None:
        pred = rf.predict(A_test)
        delta = pred - Cl_test
    
        print('\n  ================================================================================')
        print('  \033[1m Random Forest \033[0m - Prediction')
        print('  ================================================================================')
        print('   Real class\t| Predicted class\t| Delta')
        print('  --------------------------------------------------------------------------------')
        for i in range(len(pred)):
            print("   {0:.2f}\t| {1:.2f}\t\t| {2:.2f}".format(Cl_test[i], pred[i], delta[i]))
        print('  --------------------------------------------------------------------------------')
        print('   Average Delta: {0:.2f}, StDev = {1:.2f}'.format(mean(delta), stdev(delta)))
        print('   R^2: {0:.4f}'.format(rf.score(A_test, Cl_test)))
        print('  --------------------------------------------------------------------------------\n')
        print('  Scikit-learn v.',str(sklearn.__version__),'\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

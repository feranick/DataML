#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* DataML Decision Forests - Classifier and Regressor
* version: 2025.12.03.01
* Uses: sklearn, tabpfn
* By: Nicola Ferralis <feranick@hotmail.com>
*****************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, getopt, time, configparser
import platform, pickle, h5py, csv, glob, math, ast
from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def DataML_DF():
    main()

#************************************
# Parameters
#************************************
class Conf():
    def __init__(self):

        #################################
        ### Types of estimators:
        ### Set using: typeDF
        ### - GradientBoosting (default)
        ### - RandomForest
        ### - HistGradientBoosting
        ### - DecisionTree
        ### - TabPFN (experimental)
        #################################
        
        ###################################
        ### Types of optimization scoring:
        ### Set using:
        ### optScoringR for Regression
        ### optScoringC for Classification
        ### - neg_mean_absolute_error (default)
        ### - r2
        ### - accuracy
        #################################

        self.appName = "DataML_DF"
        confFileName = "DataML_DF.ini"
        self.configFile = os.path.join(os.getcwd(),confFileName)
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
        
        self.modelNameRoot = "model_DF_"
        self.modelName = self.modelNameRoot + self.typeDF + self.mode + ".pkl"
        self.summaryFileName = self.modelNameRoot + self.typeDF + self.mode + ".csv"
        
        self.tb_directory = "model_DF"
        self.model_name = self.model_directory+self.modelNameRoot
        
        self.model_le = self.model_directory+"model_le.pkl"
        self.model_scaling = self.model_directory+"model_scaling.pkl"
        self.model_pca = self.model_directory+"model_encoder.pkl"
        self.norm_file = self.model_directory+"norm_file.pkl"
        
        self.optParFile = "opt_parameters.txt"
                    
        self.rescaleForPCA = False
        if self.regressor:
            self.optScoring = self.optScoringR
        else:
            self.optScoring = self.optScoringC
        
        self.verbose = 1
                    
    def datamlDef(self):
        self.conf['Parameters'] = {
            'typeDF' : 'GradientBoosting',
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
            'normalizeLabel' : False,
            'runDimRedFlag' : False,
            'typeDimRed' : 'SparsePCA',
            'numDimRedComp' : 3,
            'plotFeatImportance' : False,
            'optimizeParameters' : False,
            'optScoringR' : 'neg_mean_absolute_error',
            'optScoringC' : 'accuracy',
            'featureReduction' : False,
            'minNumFeatures' : 4,
            }
    
    def sysDef(self):
        self.conf['System'] = {
            'random_state' : 1,
            'n_jobs' : -1,
            'saveAsTxt' : True,
            }
    
    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.datamlPar = self.conf['Parameters']
            self.sysPar = self.conf['System']
    
            self.typeDF = self.conf.get('Parameters','typeDF')
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
            self.normalizeLabel = self.conf.getboolean('Parameters','normalizeLabel')
            self.runDimRedFlag = self.conf.getboolean('Parameters','runDimRedFlag')
            self.typeDimRed = self.conf.get('Parameters','typeDimRed')
            self.numDimRedComp = self.conf.getint('Parameters','numDimRedComp')
            self.plotFeatImportance = self.conf.getboolean('Parameters','plotFeatImportance')
            self.optimizeParameters = self.conf.getboolean('Parameters','optimizeParameters')
            self.optScoringR = self.conf.get('Parameters','optScoringR')
            self.optScoringC = self.conf.get('Parameters','optScoringC')
            self.featureReduction = self.conf.getboolean('Parameters','featureReduction')
            self.minNumFeatures = self.conf.getint('Parameters','minNumFeatures')
            self.random_state = ast.literal_eval(self.sysPar['random_state'])
            self.n_jobs = self.conf.getint('System','n_jobs')
            self.saveAsTxt = self.conf.getboolean('System','saveAsTxt')
            
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
            
    # update configuration file
    def updateConfig(self, section, par, value):
        if self.conf.has_option(section, par) is True:
            self.conf.set(section, par, value)
            try:
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
            "trpcbgvoh:", ["train", "reduce", "predict", "csv", "batch", "generate", "validbatch", "opt", "help"])
    except:
        usage()
        sys.exit(2)

    if opts == []:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o in ("-t" , "--train"):
            try:
                dP.updateConfig('Parameters','featureReduction','False')
                if len(sys.argv) == 3:
                    train(sys.argv[2], None)
                else:
                    train(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)
            
        if o in ("-r" , "--reduce"):
            try:
                dP.updateConfig('Parameters','featureReduction','True')
                dP.updateConfig('Parameters','minNumFeatures',sys.argv[2])
            
                if len(sys.argv) == 4:
                    train(sys.argv[3], None)
                else:
                    train(sys.argv[3], sys.argv[4])
            except:
                usage()
                sys.exit(2)

        if o in ("-p" , "--predict"):
            try:
                predict(sys.argv[2])
            except:
                usage()
                sys.exit(2)
                
        if o in ("-c" , "--csv"):
            try:
                csvPredict(sys.argv[2])
            except:
                usage()
                sys.exit(2)

        if o in ("-b" , "--batch"):
            try:
                batchPredict(sys.argv[2])
            except:
                usage()
                sys.exit(2)
                
        if o in ("-g" , "--generate"):
            #try:
            genMissingData(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
            #except:
            #    usage()
            #    sys.exit(2)
            
        if o in ("-v" , "--validbatch"):
            try:
                validBatchPredict(sys.argv[2])
            except:
                usage()
                sys.exit(2)
            
        if o in ["-o" , "--opt"]:
            try:
                dP.updateConfig('Parameters','optimizeParameters','True')
                makeOptParameters(dP, sys.argv[2])
                if len(sys.argv) == 4:
                    train(sys.argv[3], None)
                else:
                    train(sys.argv[3], sys.argv[4])
            except:
                usage()
                sys.exit(2)

    total_time = time.perf_counter() - start_time
    print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

#************************************
# Training
#************************************
def train(learnFile, testFile):
    dP = Conf()
   
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from statistics import mean, stdev
    from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
    
    learnFileRoot = os.path.splitext(learnFile)[0]

    En, A, Cl, _ = readLearnFile(learnFile, True, dP)
    if testFile is None:
        totA = A
        totCl = Cl
        A, Cl, A_test, Cl_test, _ = formatSubset(A, Cl, dP.cv_split)
    else:
        En_test, A_test, Cl_test, _ = readLearnFile(testFile, False, dP)
        totA = np.vstack((A, A_test))
        totCl = np.append(Cl, Cl_test)
        
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
                
        le = MultiClassReductor(dP)
        le.fit(np.unique(totCl, axis=0))
        le.save()
        Cl2 = le.transform(Cl)
        
        print("  Number unique classes (training): ", np.unique(Cl).size)
        
        #if testFile is not None:
        Cl2_test = le.transform(Cl_test)
        print("  Number unique classes (validation):", np.unique(Cl_test).size)
        print("  Number unique classes (total): ", np.unique(totCl).size)

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
        if dP.typeDF == 'RandomForest':
            df = RandomForestRegressor(max_depth=dP.max_depth, n_estimators = dP.n_estimators, max_features = dP.max_features, verbose=dP.verbose, n_jobs=dP.n_jobs, random_state=dP.random_state)
        if dP.typeDF == 'HistGradientBoosting':
            df = HistGradientBoostingRegressor(max_depth=dP.max_depth, max_iter=dP.epochs, max_features = dP.max_features, verbose = dP.verbose, learning_rate=dP.l_rate, l2_regularization=0.0, random_state=dP.random_state)
        if dP.typeDF == 'GradientBoosting':
            if dP.max_features == 0:
                dP.max_features = None
            df = GradientBoostingRegressor(n_estimators = dP.epochs, max_depth=dP.max_depth, max_features = dP.max_features, verbose = dP.verbose, learning_rate=dP.l_rate, random_state=dP.random_state)
        if dP.typeDF == 'DecisionTree':
            if dP.max_features == 0:
                dP.max_features = None
            df = DecisionTreeRegressor(max_depth=dP.max_depth, max_features = dP.max_features, random_state=dP.random_state)
        if dP.typeDF == 'TabPFN':
            from tabpfn import TabPFNRegressor
            df = TabPFNRegressor()
            
    else:
        if dP.typeDF == 'RandomForest':
            df = RandomForestClassifier(max_depth=dP.max_depth, n_estimators = dP.n_estimators, max_features = dP.max_features, verbose=dP.verbose, n_jobs=dP.n_jobs, oob_score=False, random_state=dP.random_state)
        if dP.typeDF == 'HistGradientBoosting':
            df = HistGradientBoostingClassifier(max_depth=dP.max_depth, max_iter=dP.epochs, max_features = dP.max_features, verbose = dP.verbose, learning_rate=dP.l_rate, l2_regularization=0.0, random_state=dP.random_state)
        if dP.typeDF == 'GradientBoosting':
            if dP.max_features == 0:
                dP.max_features = None
            df = GradientBoostingClassifier(n_estimators = dP.epochs, max_depth=dP.max_depth, max_features = dP.max_features, verbose = dP.verbose, learning_rate=dP.l_rate, random_state=dP.random_state)
        if dP.typeDF == 'DecisionTree':
            if dP.max_features == 0:
                dP.max_features = None
            df = DecisionTreeClassifier(max_depth=dP.max_depth, max_features = dP.max_features, random_state=dP.random_state)
        if dP.typeDF == 'TabPFN':
            from tabpfn import TabPFNClassifier
            df = TabPFNClassifier()
    
    df.fit(A, Cl2)
            
    print("\n  ",dP.typeDF+dP.mode,"model saved in:", dP.modelName)
    with open(dP.modelName,'wb') as f:
        pickle.dump(df, f)
        
    if dP.normalize:
        try:
            with open(dP.norm_file, "rb") as f:
                norm = pickle.load(f)
            print("  Opening pkl file with normalization data:",dP.norm_file)
            print("  Normalizing validation file for prediction...\n")
        except:
            print("\033[1m pkl file not found \033[0m")
            sys.exit()

    if dP.regressor:
        if dP.normalize:
            pred = norm.transform_inverse(df.predict(A_test))
            Cl_test = norm.transform_inverse(Cl_test)
        else:
            pred = df.predict(A_test)
        score = mean_absolute_error(pred, Cl_test)
    else:
        if dP.normalize:
            pred = norm.transform_inverse(np.asarray(le.inverse_transform_bulk(df.predict(A_test))))
            pred_classes = norm.transform_inverse(np.asarray(le.inverse_transform_bulk(df.classes_)))
            Cl_test = norm.transform_inverse(Cl_test)
        else:
            pred = le.inverse_transform_bulk(df.predict(A_test))
            pred_classes = le.inverse_transform_bulk(df.classes_)
        proba = df.predict_proba(A_test)
        
        score = accuracy_score([int(x) for x in pred], Cl_test)

    delta = pred - Cl_test
    
    printParamDF(dP)
    
    print('\n  ================================================================================')
    print('  \033[1m',dP.typeDF,dP.mode,'\033[0m- Results')
    print('  ================================================================================')
    if dP.regressor:
        print('   Real class\t| Predicted class\t| Delta')
    else:
        print('   Real class\t| Predicted class\t| Delta\t| Probability ')
    print('  --------------------------------------------------------------------------------')
    for i in range(len(pred)):
        if dP.regressor:
            print("   {0:.4f}\t| {1:.4f}\t\t| {2:.4f}".format(Cl_test[i], pred[i], delta[i]))
        else:
            ind = np.where(proba[i]==np.max(proba[i]))[0]
            for j in range(len(ind)):
                print("   {0:.4f}\t| {1:.4f}\t\t| {2:.4f} \t\t| {3:.4f}%".format(Cl_test[i], pred[i], delta[i], 100*proba[i][ind[j]]))
            
    print('  --------------------------------------------------------------------------------')
    print('  ',dP.metric,'= {0:.4f}'.format(score))
    #print('   R^2 = {0:.4f}'.format(df.score(A_test, Cl2_test)))
    print('   R^2 = {0:.4f}'.format(r2_score(Cl_test, pred)))
    print('   StDev = {0:.2f}'.format(stdev(delta)))
    print('  ================================================================================\n')
    
    if dP.plotFeatImportance and (dP.typeDF == 'RandomForest' or dP.typeDF == 'GradientBoosting'):
        plotImportances(df, En, A_test, Cl2_test, dP)
    
    ##################################################################
    # Hyperparameter optimization
    ##################################################################
    if dP.optimizeParameters:
        print('  ========================================================')
        print('  \033[1m HyperParameters Optimization\033[0m')
        print('  ========================================================\n')
                
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, PredefinedSplit
        import json
        import pandas as pd
        
        if os.path.isfile(dP.optParFile) is False:
            makeOptParameters(dP,"1")
        
        with open(dP.optParFile) as f:
            grid = json.load(f)
            
        if dP.trainFullData:
            A_tot = A
            Cl2_tot = Cl2
            cv = 5
        else:
            A_tot = np.append(A,A_test, axis=0)
            Cl2_tot = np.append(Cl2,Cl2_test, axis=0)
            test_fold = [-1] * A.shape[0] + [0] * A_test.shape[0]
            cv = PredefinedSplit(test_fold=test_fold)
        
        #searcher = RandomizedSearchCV(estimator=df, n_jobs=dP.n_jobs, cv=cv,
        #    param_distributions=grid, scoring=dP.optScoring)
        searcher = GridSearchCV(estimator=df, n_jobs=dP.n_jobs, cv=cv,
            param_grid=grid, scoring=dP.optScoring, refit=True, verbose = 3)
        
        searchResults = searcher.fit(A_tot, Cl2_tot)
        
        print('\n  ========================================================')
        print('  \033[1m HyperParameters Optimization: Results\033[0m')
        print('  ========================================================')
        
        results = pd.DataFrame.from_dict(searchResults.cv_results_).sort_values(by='rank_test_score')
        print(results)
            
        if dP.regressor:
            print("\n Using scoring:", dP.optScoringR)
        else:
            print("\n Using scoring:", dP.optScoringC)
        
        bestParams = searchResults.best_params_
        print("\n Optimal parameters for best model:", )
        print(" ",bestParams,"\n")
    
        #print(list(bestParams.values())[0])
        #dP.random_state=list(bestParams.values())[0]
        
        for i in range(len(bestParams)):
            dP.updateConfig('Parameters',list(bestParams.keys())[i],str(list(bestParams.values())[i]))
            dP.updateConfig('System',list(bestParams.keys())[i],str(list(bestParams.values())[i]))
        
        print(" Updated parameters for best model in Data_DF.ini.")
        
        print(" Setting DataML_DF training in non-optimization mode. \n")
        dP.updateConfig('Parameters','optimizeParameters','False')
    
    ##################################################################
    # Automated feature selection and reduction
    ##################################################################
    if dP.featureReduction:
        print('  ========================================================')
        print('  \033[1m Feature Reduction \033[0m')
        print('  ========================================================\n')
        from sklearn.feature_selection import RFE
        selector = RFE(df, n_features_to_select=min(dP.minNumFeatures, A.shape[1]), step=1, verbose=2, importance_getter='auto')
        selector = selector.fit(A, Cl2)
        
        #print("\n", len(selector.support_),"features selected:", En[np.array(selector.support_, dtype=bool)])
        print(f"\n {len(selector.support_)} features selected: {En[np.array(selector.support_, dtype=bool)]}")
        
        saveRestrFeatLearnFile(dP, selector.support_, learnFile)
        saveRestrFeatLearnFile(dP, selector.support_, testFile)
        dP.updateConfig('Parameters','featureReduction','False')
        
    createConfigFileWeb(En)
    
    print(' Scikit-learn v.',str(sklearn.__version__),'\n')
    return r2_score(Cl_test, pred)
    
#************************************
# Prediction - frontend
#************************************
def predict(testFile):
    dP = Conf()
    import sklearn
    R, _ = readTestFile(testFile)
            
    with open(dP.modelName, "rb") as f:
        df = pickle.load(f)
        
    if dP.regressor:
        le = None
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
    
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
    
    pred, pred_classes, proba = getPrediction(dP, df, R, le, norm)
        
    print('\n  ================================================================================')
    print('  \033[1m',dP.typeDF,dP.mode,'\033[0m')
    print('  ================================================================================')
    if dP.regressor:
        print('   Filename\t\t| Prediction')
        print('  --------------------------------------------------------------------------------')
        print("   {0:s}\t| {1:.4f}  ".format(testFile, pred[0]))
    else:
        print('   Filename\t\t| Prediction\t| Probability')
        print('  --------------------------------------------------------------------------------')
        ind = np.where(proba[0]==np.max(proba[0]))[0]
        for j in range(len(ind)):
            print("   {0:s}\t| {1:.4f}\t| {2:.4f} ".format(testFile, pred_classes[ind[j]], 100*proba[0][ind[j]]))
        print("")
    print('  ================================================================================\n')
    print('  Scikit-learn v.',str(sklearn.__version__),'\n')

#************************************
# Prediction from CSV table
#************************************
def csvPredict(csvFile):
    dP = Conf()
    import sklearn
    import pandas as pd
    
    dataDf = pd.read_csv(csvFile)
            
    with open(dP.modelName, "rb") as f:
        df = pickle.load(f)
    
    summaryFile = np.array([['File:',csvFile,''],['DataML_DF',dP.typeDF,dP.mode]])
    
    if dP.regressor:
        summaryFile = np.vstack((summaryFile,['Sample','Predicted Value','']))
        le = None
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        summaryFile = np.vstack((summaryFile,['Sample','Predicted Value','Probability %']))
    
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
    
    print('\n  ==============================================================================')
    print('  \033[1m',dP.typeDF,dP.mode,'\033[0m')
    print('  ==============================================================================')
    
    for i in range(1,dataDf.shape[1]):
        R = np.array([dataDf.iloc[:,i].tolist()], dtype=float)
        Rorig = np.copy(R)
    
        pred, pred_classes, proba = getPrediction(dP, df, R, le, norm)

        if dP.regressor:
            print("   {0:s}\t = {1:.4f} ".format(dataDf.columns[i], pred[0]))
            summaryFile = np.vstack((summaryFile,[dataDf.columns[i],pred[0],'']))
        else:
            ind = np.where(proba[0]==np.max(proba[0]))[0]
            for j in range(len(ind)):
                print("   {0:s}\t = {1:.4f}\t ({2:.4f}%) ".format(dataDf.columns[i], pred_classes[ind[j]], 100*proba[0][ind[j]]))
                summaryFile = np.vstack((summaryFile,[dataDf.columns[i],pred_classes[ind[j]],round(100*proba[0][ind[j]],1)]))
    print('  ==============================================================================\n')
    saveSummaryFile(summaryFile, dP)
    print('  Scikit-learn v.',str(sklearn.__version__),'\n')

#************************************
# Batch Prediction
#************************************
def batchPredict(folder):
    dP = Conf()
    import sklearn
            
    with open(dP.modelName, "rb") as f:
        df = pickle.load(f)
        
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
    
    summaryFile = np.array([['Folder:',folder,''],['DataML_DF',dP.typeDF,dP.mode]])
    
    if dP.regressor:
        summaryFile = np.vstack((summaryFile,['File Name','Predicted Value','']))
        le = None
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        summaryFile = np.vstack((summaryFile,['File Name','Predicted Value','Probability %']))
     
    fileName = []
    pred = []
    proba = []
    for file in glob.glob(folder+'/*.txt'):
        R, good = readTestFile(file)
        
        if good:
            predtmp, pred_classes, probatmp = getPrediction(dP, df, R, le, norm)
            pred.append(predtmp)
            proba.append(probatmp)
            fileName.append(file)
    
    print('\n  ================================================================================')
    print('  \033[1m',dP.typeDF,dP.mode,'\033[0m')
    print('  ================================================================================')
    if dP.regressor:
        print('   Filename\t| Prediction')
        print('  --------------------------------------------------------------------------------')
        for i in range(0,len(pred)):
            print("   {0:s}\t| {1:.4f}  ".format(fileName[i], pred[i][0]))
            summaryFile = np.vstack((summaryFile,[fileName[i],pred[i][0],'']))
    else:
        print('   Filename\t\t| Prediction\t| Probability')
        print('  --------------------------------------------------------------------------------')
        for i in range(0,len(pred)):
            ind = np.where(proba[i][0]==np.max(proba[i][0]))[0]
            for j in range(len(ind)):
                print("   {0:s}\t| {1:.4f}\t| {2:.4f} ".format(fileName[i], pred_classes[ind[j]], 100*proba[i][0][ind[j]]))
                summaryFile = np.vstack((summaryFile,[fileName[i],pred_classes[ind[j]],100*proba[i][0][ind[j]]]))
        print("")
    print('  ================================================================================\n')
    
    saveSummaryFile(summaryFile, dP)
    print('  Scikit-learn v.',str(sklearn.__version__),'\n')

#************************************************************
# Batch Prediction using validation data (with real values)
#************************************************************
def validBatchPredict(testFile):
    dP = Conf()
    En_test, A_test, Cl_test, _ = readLearnFile(testFile, False, dP)
    import sklearn
            
    with open(dP.modelName, "rb") as f:
        df = pickle.load(f)
        
    if dP.runDimRedFlag:
        A_test = runPCAValid(A_test, dP)
    
    summaryFile = np.array([['File:',testFile,''],['DataML_DF',dP.typeDF,dP.mode]])
    
    if dP.regressor:
        pred = df.predict(A_test)
        summaryFile = np.vstack((summaryFile,['Index','Predicted Value','']))
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        pred = le.inverse_transform_bulk(df.predict(A_test))
        pred_classes = le.inverse_transform_bulk(df.classes_)
        proba = df.predict_proba(A_test)
        summaryFile = np.vstack((summaryFile,['Index','Predicted Value','Probability %']))
    
    print('  ================================================================================')
    print('  \033[1m',dP.typeDF,dP.mode,'\033[0m')
    print('  ================================================================================')
    if dP.regressor:
        print('   Prediction')
        print('  --------------------------------------------------------------------------------')
        for i in range(0,len(pred)):
            print("   {0:.4f}  ".format(pred[i]))
            summaryFile = np.vstack((summaryFile,[pred[i],'','']))
    else:
        print('   Prediction\t| Probability')
        print('  --------------------------------------------------------------------------------')
        for i in range(0,len(pred)):
            ind = np.where(proba[i]==np.max(proba[i]))[0]
            for j in range(len(ind)):
                print("   {0:.4f}\t| {1:.4f} ".format(pred_classes[ind[j]], 100*proba[i][ind[j]]))
                summaryFile = np.vstack((summaryFile,[i, pred_classes[ind[j]],100*proba[i][ind[j]]]))
            print("")
        
    print('  ================================================================================\n')
    
    saveSummaryFile(summaryFile, dP)
    print('  Scikit-learn v.',str(sklearn.__version__),'\n')

#************************************
# Generating missing parameters
#************************************
def genMissingData(testFile, param, min, max, step):
    from scipy.optimize import brentq
    dP = Conf()
    import sklearn
    Rtot, _ = readTestFile(testFile)
    perf = Rtot[0][-1]
    currentPar = Rtot[0][int(param)]
    R = Rtot[:,:-1]
            
    with open(dP.modelName, "rb") as f:
        df = pickle.load(f)
        
    if dP.regressor:
        le = None
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
    
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
    
    def f(x):
        R_tmp = R.copy()
        R_tmp[0][int(param)] = x
        pred, pred_classes, proba = getPrediction(dP, df, R_tmp, le, norm)
        return pred[0], pred[0] - perf
            
    print('\n  ================================================================================')
    print(f'  \033[1m Generating missing data for param #{param}; Curr. value: {currentPar}\033[0m')
    print('  ================================================================================')
    print(f'   New par #{param}\t| Pred value\t| Real value \t| Difference')
    print('  --------------------------------------------------------------------------------')
    
    minFlag = True
    for i in np.arange(int(min), int(max), int(step)):
        resid = f(i)
        if abs(resid[1]) < 1e-3:
            if minFlag is True:
                a = i-5
                minFlag = False
            print("   {0:.2f}\t| {1:.2f}\t| {2:.2f}\t| {3:.2e}".format(i, resid[0], perf, resid[1]))
        if abs(resid[1]) > 1e-3 and minFlag is False:
            b = i+5
            break
            
    #print('  ================================================================================\n')
    
    ## Using Brentq method in scipy.
    g = lambda x: f(x)[1]
    try:
        print('\n  ================================================================================')
        print(f'  \033[1m Running Brentq search for parameter #{param} from {a} to {b}\033[0m ')
        print('  ================================================================================')
        x1 = brentq(g, a, b)
    
        # Verification
        y_check = f(x1)[0]

        print(f'   Found optimal parameter value for #{param}: \033[1m {x1:.6f}\033[0m ')
        print(f'   Predicted value using the optimal parameter #{param}: {y_check:.6f}')
        print(f'   Target real value: {perf}')
        print(f'   Difference: {abs(y_check - perf):.2e}')
        print('  ================================================================================\n')

    except:
        print('\n  No Automated search is possible. Try to broaden the min and max value in your search.\n\n')
    
    print('  Scikit-learn v.',str(sklearn.__version__),'\n')

#***********************************************************
# Save Plots with the model importance
#************************************************************
def plotImportances(df, En, A_test, Cl_test, dP):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.inspection import permutation_importance

    #feature_names = [f"feature {i}" for i in range(A_test.shape[1])]
    feature_names = En2Par(En)
    importances = df.feature_importances_
    forest_importances1 = pd.Series(importances, index=feature_names)
    
    fig, ax = plt.subplots()
    if dP.typeDF == "RandomForests":
        std = np.std([tree.feature_importances_ for tree in df.estimators_], axis=0)
        forest_importances1.plot.bar(yerr=std, ax=ax)
    else:
        forest_importances1.plot.bar(ax=ax)
    ax.set_title("Feature importances using mean decrease in impurity")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    fig.savefig('model_'+dP.typeDF+dP.mode+'_importances_MDI' + '.png', dpi = 160, format = 'png')  # Save plot
    
    result = permutation_importance(
        df, A_test, Cl_test, n_repeats=100, random_state=42, n_jobs=2)
            
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
                
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    
    fig.savefig('model_'+dP.typeDF+dP.mode+'_importances_Perm' + '.png', dpi = 160, format = 'png')  # Save plot
    
    print("  Feature importances plots saved\n")

# Transform En values into a sequence of m1,m2, etc parameters
def En2Par(En):
    return ["m" + str(x) for x in En.astype(int)]

# Create config.txt for web version with relevant parameters to the model
def createConfigFileWeb(En):
    data = En2Par(En)
    with open("config.txt", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(data)
    print("  List of used parameters saved in config.txt\n ")

#************************************
# Make Optimization Parameter File
#************************************
def makeOptParameters(dP, ind):
    import json
    flag = True
    if ind == "1":
        grid = {"random_state": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            "max_depth": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
    elif ind == "2":
        grid = {"n_estimators": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            "max_features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
    elif ind == "3":
        grid = {"random_state": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            "max_depth": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            "n_estimators": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            "max_features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
    else:
        flag = False
    
    if flag:
        print(" Grid:",grid)
        with open(dP.optParFile, 'w') as json_file:
            json.dump(grid, json_file)
        print("\n Created: ",dP.optParFile)
    else:
        print("\n using existing file: ",dP.optParFile)
        
    print(" Next training wtih DataML_DF will be done in optimization mode. \n")
    dP.updateConfig('Parameters','optimizeParameters','True')
    
#************************************
# Lists the program usage
#************************************
def usage():
    print('\n Usage:\n')
    print(' Train (Random cross validation):')
    print('  DataML_DF -t <learningFile>\n')
    print(' Train (with external validation):')
    print('  DataML_DF -t <learningFile> <validationFile> \n')
    print(' Train and feature reduce (Random cross validation):')
    print('  DataML_DF -r <num features> <learningFile>\n')
    print(' Train and feature reduce (with external validation):')
    print('  DataML_DF -r <num features> <learningFile> <validationFile> \n')
    print(' Predict:')
    print('  DataML_DF -p <testFile>\n')
    print(' Batch predict:')
    print('  DataML_DF -b <folder>\n')
    print(' Batch predict from CSV file for multiple samples (DataML_DF only):')
    print('  DataML_DF -c <testfile.csv>\n')
    print(' Batch predict on validation data in single file:')
    print('  DataML_DF -v <singleValidationFile>\n')
    print(' Generate optimal values for a single parameter given the target value of the prediction:')
    print('  DataML_DF -g <paramFile> <paramToSearch> <minValue> <maxValue> <step>\n')
    
    print('  DataML_DF -v <singleValidationFile>\n')
    
    print(' Generate new training set using normal or diffused randomization on each feature:')
    print('  DataML_DF -g <learningFile> <pkl normalization file>')
    print('\n Types of estimators:')
    print('  - RandomForest')
    print('  - HistGradientBoosting')
    print('  - GradientBoosting')
    print('  - DecisionTree')
    print('\n Run hyperparameter optimization (Random cross validation):')
    print('  DataML_DF -o <type of optimization> <learningFile>')
    print('\n Run hyperparameter optimization (with external validation):')
    print('  DataML_DF -o <type of optimization> <learningFile> <validFile>')
    print('\n Types of optimization:')
    print('  1 - random_state, max_depth')
    print('  2 - n_estimators, max_features')
    print('  3 - random_state, max_depth, n_estimators, max_features\n')
    print('  else - custom file')
    
    print(' Requires python 3.x. Not compatible with python 2.x\n')
    
#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* DataML Classifier and Regressor
* v2024.09.27.1
* Uses: TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, getopt, time, configparser
import platform, pickle, h5py, csv, glob, math
from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def DataML():
    main()

#************************************
# Parameters
#************************************
class Conf():
    def __init__(self):
        confFileName = "DataML.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "./"
        if self.regressor:
            self.modelName = "model_regressor.h5"
            self.summaryFileName = "summary_regressor.csv"
            self.model_png = self.model_directory+"/model_regressor_MLP.png"
        else:
            self.modelName = "model_classifier.h5"
            self.summaryFileName = "summary_classifier.csv"
            self.model_png = self.model_directory+"/model_classifier_MLP.png"
        
        self.tb_directory = "model_MLP"
        self.model_name = self.model_directory+self.modelName
        self.model_le = self.model_directory+"model_le.pkl"
        
        self.optParFile = "opt_parameters.txt"
            
        if platform.system() == 'Linux':
            self.edgeTPUSharedLib = "libedgetpu.so.1"
        if platform.system() == 'Darwin':
            self.edgeTPUSharedLib = "libedgetpu.1.dylib"
        if platform.system() == 'Windows':
            self.edgeTPUSharedLib = "edgetpu.dll"
            
    def datamlDef(self):
        self.conf['Parameters'] = {
            'regressor' : False,
            'l_rate' : 0.001,
            'l_rdecay' : 1e-4,
            'HL' : [20,30,40,50,60,70],
            'drop' : 0,
            'l2' : 1e-4,
            'epochs' : 100,
            'cv_split' : 0.01,
            'fullSizeBatch' : True,
            'batch_size' : 64,
            'numLabels' : 1,
            'plotWeightsFlag' : False,
            'optimizeParameters' : False,
            'stopAtBest' : False,
            'saveBestModel' : False,
            'metricBestModelR' : 'val_mae',
            'metricBestModelC' : 'val_accuracy',
            }
    def sysDef(self):
        self.conf['System'] = {
            'kerasVersion' : 2,
            'fixTFseed' : True,
            'makeQuantizedTFlite' : True,
            'useTFlitePred' : False,
            'TFliteRuntime' : False,
            'runCoralEdge' : False,
            #'setMaxMem' : False,   # TensorFlow 2.0
            #'maxMem' : 4096,       # TensorFlow 2.0
            }

    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.datamlDef = self.conf['Parameters']
            self.sysDef = self.conf['System']
        
            self.regressor = self.conf.getboolean('Parameters','regressor')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.l_rdecay = self.conf.getfloat('Parameters','l_rdecay')
            self.HL = eval(self.datamlDef['HL'])
            self.drop = self.conf.getfloat('Parameters','drop')
            self.l2 = self.conf.getfloat('Parameters','l2')
            self.epochs = self.conf.getint('Parameters','epochs')
            self.cv_split = self.conf.getfloat('Parameters','cv_split')
            self.fullSizeBatch = self.conf.getboolean('Parameters','fullSizeBatch')
            self.batch_size = self.conf.getint('Parameters','batch_size')
            self.numLabels = self.conf.getint('Parameters','numLabels')
            self.plotWeightsFlag = self.conf.getboolean('Parameters','plotWeightsFlag')
            self.optimizeParameters = self.conf.getboolean('Parameters','optimizeParameters')
            self.stopAtBest = self.conf.getboolean('Parameters','stopAtBest')
            self.saveBestModel = self.conf.getboolean('Parameters','saveBestModel')
            self.metricBestModelR = self.conf.get('Parameters','metricBestModelR')
            self.metricBestModelC = self.conf.get('Parameters','metricBestModelC')
            
            self.kerasVersion = self.conf.getint('System','kerasVersion')
            self.fixTFseed = self.conf.getboolean('System','fixTFseed')
            self.makeQuantizedTFlite = self.conf.getboolean('System','makeQuantizedTFlite')
            self.useTFlitePred = self.conf.getboolean('System','useTFlitePred')
            self.TFliteRuntime = self.conf.getboolean('System','TFliteRuntime')
            self.runCoralEdge = self.conf.getboolean('System','runCoralEdge')
            #self.setMaxMem = self.conf.getboolean('System','setMaxMem')     # TensorFlow 2.0
            #self.maxMem = self.conf.getint('System','maxMem')   # TensorFlow 2.0
                        
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
                                   "tpbvloh:", ["train", "predict", "batch", "validbatch","lite", "opt", "help"])
    except:
        usage()
        sys.exit(2)

    if opts == []:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o in ("-t" , "--train"):
            try:
                if len(sys.argv)<4:
                    train(sys.argv[2], None, None)
                else:
                    if len(sys.argv)<5:
                        train(sys.argv[2], sys.argv[3], None)
                    else:
                        train(sys.argv[2], sys.argv[3], sys.argv[4])
            except:
                usage()
                sys.exit(2)

        if o in ("-p" , "--predict"):
            try:
                if len(sys.argv)<4:
                    predict(sys.argv[2], None)
                else:
                    predict(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)

        if o in ("-b" , "--batch"):
            try:
                if len(sys.argv)<4:
                    batchPredict(sys.argv[2], None)
                else:
                    batchPredict(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)
            
        if o in ("-v" , "--validbatch"):
            try:
                if len(sys.argv)<4:
                    validBatchPredict(sys.argv[2], None)
                else:
                    validBatchPredict(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)
                
        if o in ("-l" , "--lite"):
            try:
                convertTflite(sys.argv[2])
            except:
                usage()
                sys.exit(2)
                
        if o in ["-o" , "--opt"]:
            try:
                makeOptParameters(dP)
            except:
                usage()
                sys.exit(2)

    total_time = time.perf_counter() - start_time
    print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

#************************************
# Training
#************************************
def train(learnFile, testFile, normFile):
    dP = Conf()
    import tensorflow as tf
    if checkTFVersion("2.16.0"):
        import tensorflow.keras as keras
    else:
        if dP.kerasVersion == 2:
            import tf_keras as keras
        else:
            import keras
        
    if dP.fixTFseed == True:
        tf.random.set_seed(42)

    opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1)     # Tensorflow 2.0
    conf = tf.compat.v1.ConfigProto(gpu_options=opts)  # Tensorflow 2.0
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
    #   if dP.setMaxMem:
    #       tf.config.experimental.set_virtual_device_configuration(
    #         gpus[0],
    #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=dP.maxMem)])
    
    learnFileRoot = os.path.splitext(learnFile)[0]

    En, A, Cl = readLearnFile(learnFile)
    if testFile is not None:
        En_test, A_test, Cl_test = readLearnFile(testFile)
        totA = np.vstack((A, A_test))
        totCl = np.append(Cl, Cl_test)
    else:
        totA = A
        totCl = Cl

    print("  Data size:", A.shape)
    print("  Number of learning labels: {0:d}".format(int(dP.numLabels)))
    print("  Total number of points per data:",En.size)
    if testFile is not None:
        print("\n  Testing set file:",testFile)
        print("  Training set file datapoints:", A.shape[0])
        print("  Testing set file datapoints:", A_test.shape[0])
    else:
        print("\n  Testing data set from training file:",dP.cv_split*100,"%")
        print("  Training set file datapoints:", math.floor(A.shape[0]*(1-dP.cv_split)))
        print("  Testing set datapoints:", math.ceil(A.shape[0]*dP.cv_split))
    
    if dP.regressor:
        Cl2 = np.copy(Cl)
        if testFile is not None:
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
    
        print("\n  Number unique classes (training): ", np.unique(Cl).size)
    
        if testFile is not None:
            Cl2_test = le.transform(Cl_test)
            print("  Number unique classes (validation):", np.unique(Cl_test).size)
            print("  Number unique classes (total): ", np.unique(totCl).size)
            
        print("\n  Label Encoder saved in:", dP.model_le,"\n")
        with open(dP.model_le, 'ab') as f:
            f.write(pickle.dumps(le))

        #totCl2 = keras.utils.to_categorical(totCl2, num_classes=np.unique(totCl).size)
        Cl2 = keras.utils.to_categorical(Cl2, num_classes=np.unique(totCl).size+1)
        if testFile is not None:
            Cl2_test = keras.utils.to_categorical(Cl2_test, num_classes=np.unique(totCl).size+1)

    #************************************
    # Training
    #************************************

    if dP.fullSizeBatch == True:
        dP.batch_size = A.shape[0]

    #************************************
    ### Build model
    #************************************
    def get_model():
        #************************************
        ### Define optimizer
        #************************************
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=dP.l_rate,
            decay_steps=dP.epochs,
            decay_rate=dP.l_rdecay)
        optim = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9,
                beta_2=0.999, epsilon=1e-08,
                amsgrad=False)
        
        #************************************
        ### Build model
        #************************************
        model = keras.models.Sequential()
        for i in range(len(dP.HL)):
            model.add(keras.layers.Dense(dP.HL[i],
                activation = 'relu',
                input_dim=A.shape[1],
                kernel_regularizer=keras.regularizers.l2(dP.l2)))
            model.add(keras.layers.Dropout(dP.drop))

        if dP.regressor:
            model.add(keras.layers.Dense(1))
            model.compile(loss='mse',
            optimizer=optim,
            metrics=['mae'])
        else:
            model.add(keras.layers.Dense(np.unique(totCl).size+1, activation = 'softmax'))
            model.compile(loss='categorical_crossentropy',
                optimizer=optim,
                metrics=['accuracy'])
        return model
        
    #model = get_model(learnRate = dP.l_rate, decay = dP.l_rdecay, layers = dP.HL, l2 = dP.l2, dropout = dP.drop)
    model = get_model()

    tbLog = keras.callbacks.TensorBoard(log_dir=dP.tb_directory, histogram_freq=120,
            write_graph=True, write_images=False)
    
    tbLogs = [tbLog]
    if dP.stopAtBest == True:
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
        tbLogs.append(es)
    if dP.saveBestModel == True:
        if dP.regressor:
            mc = keras.callbacks.ModelCheckpoint(dP.model_name, monitor=dP.metricBestModelR, mode='min', verbose=1, save_best_only=True)
        else:
            mc = keras.callbacks.ModelCheckpoint(dP.model_name, monitor=dP.metricBestModelC, mode='max', verbose=1, save_best_only=True)
        tbLogs.append(mc)
        
    #tbLogs = [tbLog, es, mc]
    
    if testFile is not None:
        log = model.fit(A, Cl2,
            epochs=dP.epochs,
            batch_size=dP.batch_size,
            callbacks = tbLogs,
            verbose=2,
            validation_data=(A_test, Cl2_test))
    else:
        log = model.fit(A, Cl2,
            epochs=dP.epochs,
            batch_size=dP.batch_size,
            callbacks = tbLogs,
            verbose=2,
	        validation_split=dP.cv_split)
            
    if dP.saveBestModel == False:
        if dP.kerasVersion == 2:
            model.save(dP.model_name)
        else:
            model.export(os.path.splitext(dP.model_name)[0])
    else:
        model = loadModel(dP)
        
    keras.utils.plot_model(model, to_file=dP.model_png, show_shapes=True)
    
    if dP.makeQuantizedTFlite:
        makeQuantizedTFmodel(A, dP)

    print('\n  =============================================')
    print('  \033[1m CNN\033[0m - Model Architecture')
    print('  =============================================\n')
    model.summary()

    print('\n  ========================================================')
    print('  \033[1m MLP\033[0m - Training/Validation set Configuration')
    print('  ========================================================')
    #for conf in model.get_config():
    #    print(conf,"\n")
    
    print("  Data size:", A.shape)
    print("\n  Training set file:",learnFile)
    
    if testFile is not None:
        print("  Testing set file:",testFile)
        print("\n  Training set file datapoints:", A.shape[0])
        print("  Testing set file datapoints:", A_test.shape[0])
    else:
        print("  Testing data set from training file:",dP.cv_split*100,"%")
        print("\n  Training set file datapoints:", math.floor(A.shape[0]*(1-dP.cv_split)))
        print("  Testing set datapoints:", math.ceil(A.shape[0]*dP.cv_split))
    print("\n  Number of learning labels:",dP.numLabels)
    print("  Total number of points per data:",En.size)

    loss = np.asarray(log.history['loss'])
    val_loss = np.asarray(log.history['val_loss'])
    
    if normFile is not None:
        try:
            norm_file = open(normFile, "rb")
            norm = pickle.loads(norm_file.read())
            norm_file.close()
            print("\n  Opening pkl file with normalization data:",normFile)
            print(" Normalizing validation file for prediction...")
        except:
            print("\033[1m pkl file not found \033[0m")
            return

    if dP.regressor:
        mae = np.asarray(log.history['mae'])
        val_mae = np.asarray(log.history['val_mae'])
            
        printParam()
        print('\n  ==========================================================')
        print('  \033[1m MLP - Regressor\033[0m - Training Summary')
        print('  ==========================================================')
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(loss), np.amin(loss), loss[-1]))
        print("  \033[1mMean Abs Err\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(mae), np.amin(mae), mae[-1]))
        print('\n  ==========================================================')
        print('  \033[1m MLP - Regressor \033[0m - Validation Summary')
        print('  ========================================================')
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(val_loss), np.amin(val_loss), val_loss[-1]))
        print("  \033[1mMean Abs Err\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(val_mae), np.amin(val_mae), val_mae[-1]))
        if dP.saveBestModel:
            if dP.metricBestModelR == 'mae':
                if testFile:
                    score = model.evaluate(A_test, Cl_test, batch_size=dP.batch_size, verbose = 0)
                    print("  \033[1mSaved model with validation MAE:\033[0m: {0:.4f}".format(score[1]))
                print("  \033[1mSaved model with min training MAE:\033[0m {0:.4f}\n".format(np.amin(mae)))
            if dP.metricBestModelR == 'val_mae':
                print("  \033[1mSaved model with validation MAE:\033[0m {0:.4f}\n".format(np.amin(val_mae)))
            else:
                pass
        if testFile:
            predictions = model.predict(A_test)
        
            print('  ===========================================================================')
            print("  Real value | Predicted value | val_loss | val_mean_abs_err | % deviation ")
            print("  ---------------------------------------------------------------------------")
            for i in range(0,len(predictions)):
                score = model.evaluate(np.array([A_test[i]]), np.array([Cl_test[i]]), batch_size=dP.batch_size, verbose = 0)
                if normFile is not None:
                    print("  {0:.3f} ({1:.3f})  |  {2:.3f} ({3:.3f})  | {4:.4f}  |  {5:.4f} | {6:.2f}".format(norm.transform_inverse_single(Cl2_test[i]),
                        Cl2_test[i], norm.transform_inverse_single(predictions[i][0]), predictions[i][0], score[0], score[1], 100*score[1]/norm.transform_inverse_single(Cl2_test[i])))
                else:
                    print("  {0:.3f}\t| {1:.3f}\t| {2:.4f}\t| {3:.4f}\t| {4:.1f}".format(Cl2_test[i],
                        predictions[i][0], score[0], score[1], 100*score[1]/Cl2_test[i]))
            print('\n  ========================================================================  \n')
    else:
        accuracy = np.asarray(log.history['accuracy'])
        val_acc = np.asarray(log.history['val_accuracy'])
        
        print("  Number unique classes (training): ", np.unique(Cl).size)
        if testFile is not None:
            Cl2_test = le.transform(Cl_test)
            print("  Number unique classes (validation):", np.unique(Cl_test).size)
            print("  Number unique classes (total): ", np.unique(totCl).size)
        printParam()
        print('\n  ========================================================')
        print('  \033[1m MLP - Classifier \033[0m - Training Summary')
        print('  ========================================================')
        print("\n  \033[1mAccuracy\033[0m - Average: {0:.2f}%; Max: {1:.2f}%; Last: {2:.2f}%".format(100*np.average(accuracy),
            100*np.amax(accuracy), 100*accuracy[-1]))
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(loss), np.amin(loss), loss[-1]))
        print('\n\n  ========================================================')
        print('  \033[1m MLP - Classifier \033[0m - Validation Summary')
        print('  ========================================================')
        print("\n  \033[1mAccuracy\033[0m - Average: {0:.2f}%; Max: {1:.2f}%; Last: {2:.2f}%".format(100*np.average(val_acc),
        100*np.amax(val_acc), 100*val_acc[-1]))
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(val_loss), np.amin(val_loss), val_loss[-1]))
        if dP.saveBestModel:
            if dP.metricBestModelC == 'accuracy':
                print("  \033[1mSaved model with training accuracy:\033[0m {0:.4f}".format(100*np.amax(accuracy)))
            if dP.metricBestModelC == 'val_acc':
                print("  \033[1mSaved model with validation accuracy:\033[0m {0:.4f}\n".format(100*np.amax(val_acc)))
            else:
                pass

        if testFile:
            predictions = model.predict(A_test)
            print('  ===========================================================')
            print("  Real class\t| Predicted class\t| Probability")
            print("  ---------------------------------------------------")

            for i in range(predictions.shape[0]):
                predClass = np.argmax(predictions[i])
                predProb = round(100*predictions[i][predClass],2)
                predValue = le.inverse_transform([predClass])[0]
                realValue = Cl_test[i]
                if normFile is not None:
                    print("  {0:.2f} ({1:.2f})  |  {2:.2f} ({3:.2f})  |  {4:.2f}".format(norm.transform_inverse_single(realValue),
                        realValue, norm.transform_inverse_single(predValue), predValue, predProb))
                else:
                    print("  {0:.2f}\t\t| {1:.2f}\t\t\t| {2:.2f}".format(realValue, predValue, predProb))
            #print("\n  Validation - Loss: {0:.2f}; accuracy: {1:.2f}%".format(score[0], 100*score[1]))
            print('\n  ==========================================================\n')

    if dP.plotWeightsFlag == True:
        plotWeights(dP, En, A, model)
    
    getTFVersion(dP)
    
    ##################################################################
    # Hyperparameter optimization
    ##################################################################
    
    if dP.optimizeParameters:
        print('  ========================================================')
        print('  \033[1m HyperParameters Optimization\033[0m')
        print('  ========================================================\n')
    
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
        import json
        
        with open(dP.optParFile) as f:
            grid = json.load(f)
        '''
        grid = {
            "learnRate" : [1e-2, 1e-3, 1e-4],
            "l2" : [1e-3, 1e-4, 1e-5],
            "decay" : [1e-3, 1e-4, 1e-5],
            #layers,
            "dropout" : [0,0.1,0.2,0.3,0.4],
            "batch_size" : [16, 32, 64, 128, 256],
            #"epochs" : [300,400,500],
            }
        '''
        
        if dP.regressor:
            model2 = KerasRegressor(build_fn=get_model, verbose=0)
            #scoring = "neg_mean_absolute_error"
            scoring = "r2"
        else:
            model2 = KerasClassifier(build_fn=get_model, verbose=0)
            scoring = "accuracy"
            
        numGPUs = len(tf.config.list_physical_devices('GPU'))
        if numGPUs == 0:
            import multiprocessing as mp
            n_jobs = mp.cpu_count() - 1
            print(" Running paramter optimization using:",n_jobs,"CPUs\n")
        else:
            n_jobs = numGPUs
            print(" Running paramter optimization using:",n_jobs,"GPUs\n")
                
        #searcher = RandomizedSearchCV(estimator=model2, n_jobs=n_jobs, cv=3,
        #    param_distributions=grid, scoring=scoring)
        searcher = GridSearchCV(estimator=model2, n_jobs=n_jobs, cv=3,
            param_grid=grid)
        
        searchResults = searcher.fit(A, Cl2)
    
        if dP.regressor:
            #results = searchResults.cv_results_
            print(" Optimal parameters for best model: ")
            bestParams = searchResults.best_params_
            print(bestParams)
            #bestModel = searchResults.best_estimator_
            #accuracy = bestModel.score(A_test, Cl2_test)
            #print("Accuracy:", accuracy)
            #best_score = searchResults.best_score_
            #print("Best_score:", best_score)
            #print(bestModel.predict(A_test), Cl2_test)
            #print(bestModel.predict(A_test) - Cl2_test)
        
        else:
            print(" Optimal parameters for best model: ")
            bestScore = searchResults.best_score_
            bestParams = searchResults.best_params_
            print(bestParams)
            print(" Best score is {:.2f} using {}".format(bestScore))
            # extract the best model, make predictions on our data, and show a
            # classification report
            print(" Evaluating the best model...")
            bestModel = searchResults.best_estimator_
            accuracy = bestModel.score(A_test, Cl2_test)
            print(" Accuracy: {:.2f}%".format(accuracy))
    
#************************************
# Prediction
#************************************
def predict(testFile, normFile):
    dP = Conf()
    R, _ = readTestFile(testFile)

    if normFile is not None:
        try:
            norm_file = open(normFile, "rb")
            norm = pickle.loads(norm_file.read())
            norm_file.close()
            print("  Opening pkl file with normalization data:",normFile)
            print("  Normalizing validation file for prediction...\n")
            R = norm.transform_valid_data(R)
        except:
            print("\033[1m pkl file not found \033[0m")
            return
    
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
        le_file = open(dP.model_le, "rb")
        le = pickle.loads(le_file.read())
        le_file.close()
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
    #En_test, A_test, Cl_test = readLearnFile(testFile)
    
    model = loadModel(dP)

    if normFile is not None:
        try:
            norm_file = open(normFile, "rb")
            norm = pickle.loads(norm_file.read())
            norm_file.close()
            print("  Opening pkl file with normalization data:",normFile,"\n")
        except:
            print("\033[1m" + " pkl file not found \n" + "\033[0m")
            return
            
    fileName = []
    for file in glob.glob(folder+'/*.txt'):
        R, good = readTestFile(file)
        if  normFile is not None:
            R = norm.transform_valid_data(R)
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
        le_file = open(dP.model_le, "rb")
        le = pickle.loads(le_file.read())
        le_file.close()
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
    En_test, A_test, Cl_test = readLearnFile(testFile)
    model = loadModel(dP)

    if normFile is not None:
        try:
            norm_file = open(normFile, "rb")
            norm = pickle.loads(norm_file.read())
            norm_file.close()
            print("  Opening pkl file with normalization data:",normFile,"\n")
        except:
            print("\033[1m" + " pkl file not found \n" + "\033[0m")
            return

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
        le_file = open(dP.model_le, "rb")
        le = pickle.loads(le_file.read())
        le_file.close()
        predictions, _ = getPredictions(A_test, model,dP)
        #predictions = model.predict(A_test)
        print('  ========================================================')
        print('  \033[1m MLP - Classifier\033[0m - Batch Prediction')
        print('  ========================================================')
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

#****************************************************
# Convert model to quantized TFlite
#****************************************************
def convertTflite(learnFile):
    dP = Conf()
    dP.useTFlitePred = False
    dP.TFliteRuntime = False
    dP.runCoralEdge = False
    from pkg_resources import parse_version
    import tensorflow as tf
    if parse_version(tf.version.VERSION) < parse_version('2.0.0'):
        tf.compat.v1.enable_eager_execution()
    learnFileRoot = os.path.splitext(learnFile)[0]
    En, A, Cl = readLearnFile(learnFile)
    model = loadModel(dP)
    makeQuantizedTFmodel(A, dP)
    
#************************************
# Open Training Data
#************************************
def readLearnFile(learnFile):
    print("  Opening training file:",learnFile)
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
        print("\033[1m Training file not found\033[0m")
        return

    dP = Conf()
    En = M[0,dP.numLabels:]
    A = M[1:,dP.numLabels:]
    if dP.numLabels == 1:
        Cl = M[1:,0]
    else:
        Cl = M[1:,[0,dP.numLabels-1]]

    return En, A, Cl

#************************************
# Open Testing Data
#************************************
def readTestFile(testFile):
    try:
        with open(testFile, 'r') as f:
            print('\n  Opening sample data for prediction:\n  ',testFile)
            Rtot = np.loadtxt(f, unpack =True)
        R=np.array([Rtot[1,:]])
        Rx=np.array([Rtot[0,:]])
    except:
        print("\033[1m\n File not found or corrupt\033[0m\n")
        return 0, False
    return R, True

#************************************
# Print NN Info
#************************************
def printParam():
    dP = Conf()
    print('\n  ================================================')
    print('  \033[1m MLP\033[0m - Parameters')
    print('  ================================================')
    print('  Optimizer:','Adam',
                '\n  Hidden layers:', dP.HL,
                '\n  Activation function:','relu',
                '\n  L2:',dP.l2,
                '\n  Dropout:', dP.drop,
                '\n  Learning rate:', dP.l_rate,
                '\n  Learning decay rate:', dP.l_rdecay)
    if dP.fullSizeBatch == True:
        print('  Batch size: full')
    else:
        print('  Batch size:', dP.batch_size)
    print('  Epochs:',dP.epochs)
    print('  Number of labels:', dP.numLabels)
    print('  Stop at Best Model based on validation:', dP.stopAtBest)
    print('  Save Best Model based on validation:', dP.saveBestModel)
    if dP.regressor:
        print('  Metric for Best Regression Model:', dP.metricBestModelR)
    else:
        print('  Metric for Best Classifier Model:', dP.metricBestModelC)
    #print('  ================================================\n')

#************************************
# Open Learning Data
#************************************
def plotWeights(dP, En, A, model):
    import matplotlib.pyplot as plt
    plt.figure(tight_layout=True)
    #plotInd = 711
    plotInd = (len(dP.HL)+2)*100+11
    for layer in model.layers:
        try:
            w_layer = layer.get_weights()[0]
            ax = plt.subplot(plotInd)
            newX = np.arange(En[0], En[-1], (En[-1]-En[0])/w_layer.shape[0])
            plt.plot(En, np.interp(En, newX, w_layer[:,0]), label=layer.get_config()['name'])
            plt.legend(loc='upper right')
            plt.setp(ax.get_xticklabels(), visible=False)
            plotInd +=1
        except:
            pass

    ax1 = plt.subplot(plotInd)
    ax1.plot(En, A[0], label='Sample data')

    plt.xlabel('Raman shift [1/cm]')
    plt.legend(loc='upper right')
    plt.savefig('model_MLP_weights' + '.png', dpi = 160, format = 'png')  # Save plot

#************************************
# Make Optimization Parameter File
#************************************
def makeOptParameters(dP):
    import json
    grid = {"learnRate": [0.01, 0.001, 0.0001], "l2": [0.001, 0.0001, 1e-05], "decay": [0.001, 0.0001, 1e-05], "dropout": [0, 0.1, 0.2, 0.3, 0.4], "batch_size": [16, 32, 64, 128, 256], "epochs": [300, 400, 500]}
    with open(dP.optParFile, 'w') as json_file:
        json.dump(grid, json_file)
    print(" Created: ",dP.optParFile,"\n")

#************************************
# Lists the program usage
#************************************
def usage():
    print('\n Usage:\n')
    print(' Train (Random cross validation):')
    print('  python3 DataML.py -t <learningFile>\n')
    print(' Train (with external validation):')
    print('  python3 DataML.py -t <learningFile> <validationFile> \n')
    print(' Train (with external validation, with labels normalized with pkl file):')
    print('  python3 DataML.py -t <learningFile> <validationFile> <pkl normalization file>\n')
    print(' Predict (no label normalization used):')
    print('  python3 DataML.py -p <testFile>\n')
    print(' Predict (labels normalized with pkl file):')
    print('  python3 DataML.py -p <testFile> <pkl normalization file>\n')
    print(' Batch predict (no label normalization used):')
    print('  python3 DataML.py -b <folder>\n')
    print(' Batch predict (labels normalized with pkl file):')
    print('  python3 DataML.py -b <folder> <pkl normalization file>\n')
    print(' Batch predict on validation data in single file (no label normalization used):')
    print('  python3 DataML.py -v <singleValidationFile>\n')
    print(' Batch predict on validation data in single file (labels normalized with pkl file):')
    print('  python3 DataML.py -v <singleValidationFile> <pkl normalization file>\n')
    print(' Convert model to quantized tflite:')
    print('  python3 SpectraKeras_CNN.py -l <learningFile>\n')
    print(' Create parameter optimization file:')
    print('  python3 SpectraKeras_CNN.py -o\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

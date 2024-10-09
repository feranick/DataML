# -*- coding: utf-8 -*-
'''
***********************************************************
* libDataML - Library for DataML
* v2024.10.09.2
* Uses: Keras, TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
import numpy as np
import scipy
import os.path, pickle, h5py

#************************************
# Normalizer
#************************************
class Normalizer(object):
    def __init__(self, M, dP):
        self.M = M
        self.normalizeLabel = dP.normalizeLabel
        self.useGeneralNormLabel = dP.useGeneralNormLabel
        self.useCustomRound = dP.useCustomRound
        self.minGeneralLabel = dP.minGeneralLabel
        self.maxGeneralLabel = dP.maxGeneralLabel
        self.YnormTo = dP.YnormTo
        self.stepNormLabel = dP.stepNormLabel
        self.saveNormalized = dP.saveNormalized
        
        self.data = np.arange(0,1,self.stepNormLabel)
        self.min = np.zeros([self.M.shape[1]])
        self.max = np.zeros([self.M.shape[1]])
    
        if self.normalizeLabel:
            if self.useGeneralNormLabel:
                self.min[0] = dP.minGeneralLabel
                self.max[0] = dP.maxGeneralLabel
            else:
                self.min[0] = np.nanmin(self.M[1:,0])
                self.max[0] = np.nanmax(self.M[1:,0])
        
        for i in range(1,M.shape[1]):
            self.min[i] = np.amin(self.M[1:,i])
            self.max[i] = np.amax(self.M[1:,i])
    
    def transform_matrix(self,y):
        Mn = np.copy(y)
        if self.normalizeLabel:
            Mn[1:,0] = np.multiply(y[1:,0] - self.min[0],
                self.YnormTo/(self.max[0] - self.min[0]))
            if self.useCustomRound:
                customData = CustomRound(self.data)
                for i in range(1,y.shape[0]):
                    Mn[i,0] = customData(Mn[i,0])
        if self.saveNormalized:
            for i in range(1,y.shape[1]):
                Mn[1:,i] = np.multiply(y[1:,i] - self.min[i],
                    self.YnormTo/(self.max[i] - self.min[i]))
        return Mn
        
    def transform_valid(self,V):
        Vn = np.copy(V)
        for i in range(0,V.shape[0]):
            Vn[i,1] = np.multiply(V[i,1] - self.min[i+1],
                self.YnormTo/(self.max[i+1] - self.min[i+1]))
        return Vn
    
    def transform_valid_data(self,V):
        Vn = np.copy(V)
        if self.saveNormalized:
            for i in range(0,V.shape[1]):
                Vn[0][i] = np.multiply(V[0][i] - self.min[i+1],
                    self.YnormTo/(self.max[i+1] - self.min[i+1]))
        return Vn
    
    def transform_inverse_single(self,v):
        vn = self.min[0] + v*(self.max[0] - self.min[0])/self.YnormTo
        return vn

    def save(self, name):
        with open(name, 'ab') as f:
            f.write(pickle.dumps(self))

#************************************
# CustomRound
#************************************
class CustomRound:
    def __init__(self,iterable):
        self.data = sorted(iterable)

    def __call__(self,x):
        from bisect import bisect_left
        data = self.data
        ndata = len(data)
        idx = bisect_left(data,x)
        if idx <= 0:
            return data[0]
        elif idx >= ndata:
            return data[ndata-1]
        x0 = data[idx-1]
        x1 = data[idx]
        if abs(x-x0) < abs(x-x1):
            return x0
        return x1

#************************************
# MultiClassReductor
#************************************
class MultiClassReductor():
    def __self__(self):
        self.name = name
    
    def fit(self,tc):
        self.totalClass = tc.tolist()
    
    def transform(self,y):
        Cl = np.zeros(y.shape[0])
        for j in range(len(y)):
            Cl[j] = self.totalClass.index(np.array(y[j]).tolist())
        return Cl
    
    def inverse_transform(self,a):
        return [self.totalClass[int(a[0])]]

    def classes_(self):
        return self.totalClass

#************************************
# Load saved models
#************************************
def loadModel(dP):
    if dP.TFliteRuntime:
        import tflite_runtime.interpreter as tflite
        # model here is intended as interpreter
        if dP.runCoralEdge:
            print(" Running on Coral Edge TPU")
            try:
                model = tflite.Interpreter(model_path=os.path.splitext(dP.model_name)[0]+'_edgetpu.tflite',
                    experimental_delegates=[tflite.load_delegate(dP.edgeTPUSharedLib,{})])
            except:
                print(" Coral Edge TPU not found. \n Please make sure it's connected and Tflite-runtime matches the TF version that is installled.")
        else:
            model = tflite.Interpreter(model_path=os.path.splitext(dP.model_name)[0]+'.tflite')
        model.allocate_tensors()
    else:
        getTFVersion(dP)
        import tensorflow as tf
        if checkTFVersion("2.16.0"):
            import tensorflow.keras as keras
        else:
            if dP.kerasVersion == 2:
                import tf_keras as keras
            else:
                import keras
        if dP.useTFlitePred:
            # model here is intended as interpreter
            model = tf.lite.Interpreter(model_path=os.path.splitext(dP.model_name)[0]+'.tflite')
            model.allocate_tensors()
        else:
            if dP.kerasVersion == 2:
                model = keras.models.load_model(dP.model_name)
            else:
                model = keras.saving.load_model(dP.model_name)
    print("  Model name:",dP.model_name)
    return model

#************************************
# Make prediction based on framework
#************************************
def getPredictions(R, model, dP):
    if dP.useTFlitePred:
        interpreter = model  #needed to keep consistency with documentation
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(R*255, dtype=np.uint8) # Disable this for TF1.x
        #input_data = np.array(R, dtype=np.float32)  # Enable this for TF2.x (not compatible with on EdgeTPU)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
    else:
        predictions = model.predict(R)
        
    probabilities = scipy.special.softmax(predictions.astype('double'))
    return predictions, probabilities

#************************************
### Create Quantized tflite model
#************************************
def makeQuantizedTFmodel(A, dP):
    import tensorflow as tf
    print("\n  Creating quantized TensorFlowLite Model...\n")
    
    A2 = tf.cast(A, tf.float32)
    A = tf.data.Dataset.from_tensor_slices((A2)).batch(1)
    
    def representative_dataset_gen():
        for input_value in A.take(100):
            yield[input_value]
            
    if dP.kerasVersion == 2:
        if checkTFVersion("2.16.0"):
            import tensorflow.keras as keras
        else:
            import tf_keras as keras
        model = keras.models.load_model(dP.model_name)
    else:
        # Previous method, TF <= 2.16.2
        #import keras
        #model = keras.layers.TFSMLayer(dP.model_name, call_endpoint='serve')
        #converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # New method
        #model = tf.saved_model.load(dP.model_name)
        #concrete_func = model.signatures['serving_default']
        #converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        # New method 2:
        import keras
        model = keras.saving.load_model(dP.model_name)
        
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()

    with open(os.path.splitext(dP.model_name)[0]+'.tflite', 'wb') as o:
        o.write(tflite_quant_model)

#************************************
# Get TensorFlow Version
#************************************
def getTFVersion(dP):
    import tensorflow as tf
    if checkTFVersion("2.16.0"):
        import tensorflow.keras as keras
        kv = "- Keras v. " + keras.__version__
    else:
        if dP.kerasVersion == 2:
            import tf_keras as keras
            kv = "- tf_keras v. " + keras.__version__
        else:
            import keras
            kv = "- Keras v. " + keras.__version__
    from packaging import version
    if dP.useTFlitePred:
        print("\n TensorFlow (Lite) v.",tf.version.VERSION,kv, "\n")
    else:
        print("\n TensorFlow v.",tf.version.VERSION,kv, "\n" )
        
def checkTFVersion(vers):
    import tensorflow as tf
    from packaging import version
    v = version.parse(tf.__version__)
    return v < version.parse(vers)
    

#****************************************************
# Convert model to quantized TFlite
#****************************************************
def convertTflite(learnFile, dP):
    dP = Conf()
    dP.useTFlitePred = False
    dP.TFliteRuntime = False
    dP.runCoralEdge = False
    from pkg_resources import parse_version
    import tensorflow as tf
    if parse_version(tf.version.VERSION) < parse_version('2.0.0'):
        tf.compat.v1.enable_eager_execution()
    learnFileRoot = os.path.splitext(learnFile)[0]
    En, A, Cl = readLearnFile(learnFile, dP)
    model = loadModel(dP)
    makeQuantizedTFmodel(A, dP)
    
#************************************
# Open Training Data
#************************************
def readLearnFile(learnFile, dP):
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
def readTestFile(testFile, dP):
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
def printParam(dP):
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
# Plot Weigths
#************************************
def plotWeights(En, A, model, dP):
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
# Autoencoder
#************************************
def runAutoencoder(learnFile, testFile, dP):
    import keras
    import tensorflow as tf
    
    class Autoencoder(keras.Model):
        def __init__(self, latent_dim, shape):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim
            self.shape = shape
            self.encoder = keras.Sequential([
                keras.layers.Flatten(),
                keras.layers.Dense(latent_dim, activation='relu'),
                ])
            self.decoder = keras.Sequential([
                keras.layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
                keras.layers.Reshape(shape)
                ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    En, A, Cl = readLearnFile(learnFile, dP)
            
    shape = A.shape[1:]
    latent_dim = 4
    autoencoder = Autoencoder(latent_dim, shape)
    autoencoder.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    
    if testFile is None:
        autoencoder.fit(A, A,
                epochs=10,
                shuffle=True,
                #validation_data=(x_test, x_test),
                validation_split=dP.cv_split
                )
    else:
        En_test, A_test, Cl_test = readLearnFile(testFile, dP)
        autoencoder.fit(A, A,
                epochs=10,
                shuffle=True,
                validation_data=(A_test, A_test),
                validation_split=dP.cv_split
                )
                
        A_test_encoded = autoencoder.encoder(A_test).numpy()
        A_test_decoded = autoencoder.decoder(A_test_encoded).numpy()
        
        print(A_test)
        print(A_test_encoded)
        print(A_test_decoded)

#********************************************************************************
# Run PCA - EXPERIMENTAL
# Transform data:
#    pca.fit(data).transform(data)
#    Loading Vectors (eigenvectors):
#    pca.components_
#    Eigenvalues:
#    pca.explained_variance_ratio
#
#********************************************************************************
def runPCA(En, Cl, A, numPCAcomponents):
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from matplotlib import cm

    customNumPCAComp = True
    showPCAPlots = True
    
    print('==========================================================================\n')
    print(' Running PCA...\n')
    print(' Number of unique target classes in training data: ' + str(np.unique(Cl).shape[0]))
    if customNumPCAComp == False:
        numPCAcomp = np.unique(Cl).shape[0]
    else:
        numPCAcomp = numPCAcomponents
    print(' Number of Principal components: ' + str(numPCAcomp) + '\n')
    pca = PCA(n_components=numPCAcomp)
    A_r = pca.fit(A).transform(A)
    
    for i in range(0,pca.components_.shape[0]):
        print(' Score PC ' + str(i) + ': ' + '{0:.0f}%'.format(pca.explained_variance_ratio_[i] * 100))
    print('')

    if showPCAPlots:
        print(' Plotting Loadings and score plots... \n')

        #***************************
        # Plotting Loadings
        #***************************
        for i in range(0,pca.components_.shape[0]):
            plt.plot(En, pca.components_[i,:], label='PC' + str(i) + ' ({0:.0f}%)'.format(pca.explained_variance_ratio_[i] * 100))
        plt.plot((En[0], En[En.shape[0]-1]), (0.0, 0.0), 'k--')
        plt.title('Loadings plot')
        plt.xlabel('Parameter')
        plt.ylabel('Principal component')
        plt.legend()
        plt.figure()
        
        
        #***************************
        # Plotting Scores
        #***************************
        if len(Cl):
            Cl_ind = np.zeros(len(Cl))
            Cl_labels = np.zeros(0)
            ind = np.zeros(np.unique(Cl).shape[0])
            for i in range(len(Cl)):
                if (np.in1d(Cl[i], Cl_labels, invert=True)):
                    Cl_labels = np.append(Cl_labels, Cl[i])

            for i in range(len(Cl)):
                Cl_ind[i] = np.where(Cl_labels == Cl[i])[0][0]
                colors = [ cm.jet(x) for x in np.linspace(0, 1, ind.shape[0]) ]

            for color, i, target_name in zip(colors, range(ind.shape[0]), Cl_labels):
                plt.scatter(A_r[Cl_ind==i,0], A_r[Cl_ind==i,1], color=color, alpha=.8, lw=2, label=target_name)

            plt.title('Score plot')
            plt.xlabel('PC 0 ({0:.0f}%)'.format(pca.explained_variance_ratio_[0] * 100))
            plt.ylabel('PC 1 ({0:.0f}%)'.format(pca.explained_variance_ratio_[1] * 100))
            plt.figure()

            plt.title('Score box plot')
            plt.xlabel('Principal Component')
            plt.ylabel('Score')
            for j in range(pca.components_.shape[0]):
                for color, i, target_name in zip(colors, range(ind.shape[0]), Cl_labels):
                    plt.scatter([j+1]*len(A_r[Cl_ind==i,j]), A_r[Cl_ind==i,j], color=color, alpha=.8, lw=2, label=target_name)
            plt.boxplot(A_r)
            plt.figure()

        #******************************
        # Plotting Scores vs Parameters
        #******************************
        for j in range(pca.components_.shape[0]):
            for color, i, target_name in zip(colors, range(ind.shape[0]), Cl_labels):
                plt.scatter(np.asarray(Cl)[Cl_ind==i], A_r[Cl_ind==i,j], color=color, alpha=.8, lw=2, label=target_name)
            plt.xlabel('Parameter')
            plt.ylabel('PC ' + str(j) + ' ({0:.0f}%)'.format(pca.explained_variance_ratio_[j] * 100))
            plt.figure()
        
        plt.show()
        


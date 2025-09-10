# -*- coding: utf-8 -*-
'''
**************************************************
* libDataML - Library for DataML/DataML_DF
* version: 2025.09.10.1
* Uses: Keras, TensorFlow, scikit-learn
* By: Nicola Ferralis <feranick@hotmail.com>
**************************************************
'''
import numpy as np
import scipy
import sys, os.path, pickle, h5py

#************************************
# Normalizer
#************************************
class Normalizer(object):
    def __init__(self, M, dP):
        self.M = M
        self.normalizeLabel = dP.normalizeLabel
        self.useCustomRound = False
        self.minGeneralLabel = 0
        self.maxGeneralLabel = 1
        self.YnormTo = 1
        self.stepNormLabel = 0.01
        self.saveNormalized = True
        self.norm_file = "norm_file.pkl"
        
        self.data = np.arange(0,1,self.stepNormLabel)
        self.min = np.zeros([self.M.shape[1]])
        self.max = np.zeros([self.M.shape[1]])
    
        if self.normalizeLabel:
            self.min[0] = np.nanmin(self.M[1:,0])
            self.max[0] = np.nanmax(self.M[1:,0])
        
        for i in range(1,M.shape[1]):
            self.min[i] = np.nanmin(self.M[1:,i])
            self.max[i] = np.nanmax(self.M[1:,i])
    
    def transform(self,y):
        Mn = np.copy(y)
        if self.normalizeLabel:
            if self.max[0] - self.min[0] == 0:
                Mn[1:,0] = (self.max[0] + self.min[0])/2
            else:
                Mn[1:,0] = np.multiply(y[1:,0] - self.min[0],
                    self.YnormTo/(self.max[0] - self.min[0]))
            if self.useCustomRound:
                customData = CustomRound(self.data)
                for i in range(1,y.shape[0]):
                    Mn[i,0] = customData(Mn[i,0])
        if self.saveNormalized:
            for i in range(1,y.shape[1]):
                if self.max[i] - self.min[i] == 0:
                    Mn[1:,i] = (self.max[i] + self.min[i])/2
                else:
                    Mn[1:,i] = np.multiply(y[1:,i] - self.min[i],
                        self.YnormTo/(self.max[i] - self.min[i]))
        return Mn
        
    def transform_valid(self,V):
        Vn = np.copy(V)
        for i in range(0,V.shape[0]):
            if self.max[i+1] - self.min[i+1] == 0:
                Vn[i,1] = (self.max[i+1] + self.min[i+1])/2
            else:
                Vn[i,1] = np.multiply(V[i,1] - self.min[i+1],
                    self.YnormTo/(self.max[i+1] - self.min[i+1]))
        return Vn
    
    def transform_valid_data(self,V):
        Vn = np.copy(V)
        if self.saveNormalized:
            for i in range(0,V.shape[1]):
                if self.max[i+1] - self.min[i+1] == 0:
                    Vn[0][i] = (self.max[i+1] - self.min[i+1])/2
                else:
                    Vn[0][i] = np.multiply(V[0][i] - self.min[i+1],
                    self.YnormTo/(self.max[i+1] - self.min[i+1]))
        return Vn
    
    # Single sample, format [[0,1,2,3,4...]]
    def transform_inverse_single(self,v):
        vn = self.min[0] + v*(self.max[0] - self.min[0])/self.YnormTo
        return vn

    # Multiple samples, format as vertical stacks of singles
    def transform_inverse(self,V):
        Vnt = []
        for i in range (V.shape[0]):
            vn = self.min[0] + V[i]*(self.max[0] - self.min[0])/self.YnormTo
            Vnt.append(vn)
        return np.array(Vnt)

    # format simular to original DataML file
    def transform_inverse_features(self,V):
        Vn = []
        for i in range (V.shape[1]):
            vn = (self.min[i] + + V[1:,i]*(self.max[i] - self.min[i])/self.YnormTo)
            Vn.append(vn)
        return np.vstack([V[0,:],np.vstack(Vn).T])

    def save(self):
        with open(self.norm_file, 'wb') as f:
            pickle.dump(self, f)

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
    def __init__(self,dP):
        self.model_le = dP.model_le

    def fit(self,tc):
        self.totalClass = tc.tolist()
    
    def transform(self,y):
        Cl = np.zeros(y.shape[0])
        for j in range(len(y)):
            Cl[j] = self.totalClass.index(np.array(y[j]).tolist())
        return Cl
    
    def inverse_transform(self,a):
        return [self.totalClass[int(a[0])]]
        
    def inverse_transform_bulk(self,a):
        inv=[]
        for i in range(len(a)):
            inv.append(self.totalClass[int(a[i])])
        return inv

    def classes_(self):
        return self.totalClass
        
    def save(self):
        with open(self.model_le, 'wb') as f:
            pickle.dump(self, f)

#************************************
# Load saved models
#************************************
def loadModel(dP):
    getTFVersion(dP)
    if dP.TFliteRuntime:
        #import tflite_runtime.interpreter as litert
        import ai_edge_litert.interpreter as litert
        # model here is intended as interpreter
        if dP.runCoralEdge:
            print(" Running on Coral Edge TPU")
            try:
                model_name = os.path.splitext(dP.model_name)[0]+'_edgetpu.tflite'
                model = litert.Interpreter(model_path=model_name,
                    experimental_delegates=[litert.load_delegate(dP.edgeTPUSharedLib,{})])
            except:
                print(" Coral Edge TPU not found. Please make sure it's connected and Tflite-runtime matches the TF version that is installled.")
        else:
            model_name = os.path.splitext(dP.model_name)[0]+'.tflite'
            model = litert.Interpreter(model_path=model_name)
        model.allocate_tensors()
    else:
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
            model_name=os.path.splitext(dP.model_name)[0]+'.tflite'
            
            try:
                # Using tf.lite (to be deprecated)
                model = tf.lite.Interpreter(model_path=model_name)
                print("  Running on tf.lite\n")
            except:
                # Using ai_edge_litert
                import ai_edge_litert.interpreter as litert
                model = litert.Interpreter(model_path=model_name)
                print("  Running on ai_edge_litert\n")
            
            model.allocate_tensors()
        else:
            model_name = dP.model_name
            if dP.kerasVersion == 2:
                model = keras.models.load_model(model_name)
            else:
                model = keras.saving.load_model(model_name)
    print("  Model name:", model_name)
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
    if dP.regressor:
        probabilities = ""
    else:
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
        if dP.TFliteRuntime:
            try:
                import tflite_runtime as litert
                print("\n  TFlite-runtime v.",litert.__version__,kv, "\n")
            except:
                import ai_edge_litert as litert
                print("\n  LiteRT v.",litert.__version__,kv, "\n")
        else:
            print("\n  TensorFlow (Lite) v.",tf.version.VERSION,kv, "\n")
    else:
        print("\n  TensorFlow v.",tf.version.VERSION,kv, "\n" )
        
def checkTFVersion(vers):
    import tensorflow as tf
    from packaging import version
    v = version.parse(tf.__version__)
    return v < version.parse(vers)
    

#****************************************************
# Convert model to quantized TFlite
#****************************************************
def convertTflite(learnFile, dP):
    dP.useTFlitePred = False
    dP.TFliteRuntime = False
    dP.runCoralEdge = False
    from pkg_resources import parse_version
    import tensorflow as tf
    if parse_version(tf.version.VERSION) < parse_version('2.0.0'):
        tf.compat.v1.enable_eager_execution()
    learnFileRoot = os.path.splitext(learnFile)[0]
    En, A, Cl, _ = readLearnFile(learnFile, dP)
    model = loadModel(dP)
    makeQuantizedTFmodel(A, dP)
    
#************************************
# Open Training Data
#************************************
def readFile(learnFile):
    print("  Opening training file:\n  ",learnFile,"\n")
    try:
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        elif os.path.splitext(learnFile)[1] == ".h5":
            with h5py.File(learnFile, 'r') as hf:
                M = hf["M"][:]
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
        return M
    except:
        print("\033[1m Training file not found\033[0m")
        sys.exit()

def readLearnFile(learnFile, initNorm, dP):
    M = readFile(learnFile)
    
    if dP.normalize:
        print("\n  Normalization of feature matrix to 1")
        if initNorm:
            print("  Normalization parameters saved in:", dP.norm_file,"\n")
            norm = Normalizer(M, dP)
            norm.save()
        else:
            try:
                with open(dP.norm_file, "rb") as f:
                    norm = pickle.load(f)
                print("  Opening pkl file with normalization data:",dP.norm_file)
            except:
                print("\033[1m pkl file not found \033[0m")
                return
        print("  Normalizing validation file for prediction...\n")
        M = norm.transform(M)
        
    En = M[0,dP.numLabels:]
    A = M[1:,dP.numLabels:]
    if dP.numLabels == 1:
        Cl = M[1:,0]
    else:
        Cl = M[1:,0:dP.numLabels]

    return En, A, Cl, M
    
#***************************************
# Save new learning Data
#***************************************
def saveLearnFile(dP, M, learnFile, tag):
    learnFileRoot = os.path.splitext(learnFile)[0]
    if dP.saveAsTxt == True:
        learnFileRoot += tag + '.txt'
        print("\n  Saving new training file (txt) in:\n  ", learnFileRoot+"\n")
        with open(learnFileRoot, 'w') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    else:
        learnFileRoot += tag + '.h5'
        print("\n Saving new training file (hdf5) in:\n  "+learnFileRoot+"\n")
        with h5py.File(learnFileRoot, 'w') as hf:
            hf.create_dataset("M",  data=M)
            
def saveRestrFeatLearnFile(dP, listSel, learnFile):
    column_mask = np.insert(np.array(listSel, dtype=bool),0,True)
    M = readFile(learnFile)
    Mred = M[:,column_mask]
    saveLearnFile(dP, M[:,column_mask], learnFile, "_"+str(Mred.shape[1]-1)+"-par")
    
#************************************
# Open Testing Data
#************************************
def readTestFile(testFile):
    try:
        with open(testFile, 'r') as f:
            print("\n  Opening sample data for prediction:",testFile)
            Rtot = np.loadtxt(f, unpack =True)
        R=np.array([Rtot[1,:]])
        Rx=np.array([Rtot[0,:]])
    except:
        print("\033[1m\n File not found or corrupt\033[0m\n")
        return 0, False
    return R, True

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

    plt.xlabel('Parameter')
    plt.legend(loc='upper right')
    plt.savefig('model_MLP_weights' + '.png', dpi = 160, format = 'png')  # Save plot
    
#********************************************************************************
# Perform PCA for feature dimensionality reduction - EXPERIMENTAL
#********************************************************************************
# Define correct value of numPCA
def prePCA(learnFile, validFile, dP):
    En, A, Cl, _ = readLearnFile(learnFile, dP)
    if dP.numDimRedComp > min(En.shape[0],Cl.shape[0]):
        numPCA = min(En.shape[0],Cl.shape[0])
    else:
        numPCA = dP.numDimRedComp
    A_encoded = runPCA(A, dP.numDimRedComp, dP)
    if dP.typeDimRed == "PCA":
        statsPCA(En, A_encoded, Cl, dP)

def runPCA(A, numDimRedComp, dP):
    import numpy as np
    from sklearn import preprocessing, decomposition
        
    #**************************************
    # Sklearn SparsePCA, PCA, TruncatedSVD
    #**************************************
    
    if dP.typeDimRed == "SparsePCA":
        spca = decomposition.SparsePCA(n_components=numDimRedComp, alpha = 0.1, verbose=2)
    if dP.typeDimRed == "PCA":
        spca = decomposition.PCA(n_components=numDimRedComp)
    if dP.typeDimRed == "TruncatedSVD":
        spca = decomposition.TruncatedSVD(n_components=numDimRedComp)
    
    #print("  Running PCA (using: "+dP.typeDimRed+")")
    print("  Number of Principal components:",str(numDimRedComp),"\n")
    
    if dP.rescaleForPCA:
        scaler = preprocessing.StandardScaler(with_mean=False)
        A_prep = scaler.fit_transform(A)
        print("  Scaling encoder saved in:", dP.model_scaling)
        with open(dP.model_scaling,'wb') as f:
            pickle.dump(scaler, f)
            
        A_encoded = spca.fit_transform(A_prep)
        A_decoded = scaler.inverse_transform(spca.inverse_transform(A_encoded))
    else:
        A_encoded = spca.fit_transform(A)
        A_decoded = spca.inverse_transform(A_encoded)
    
    print(" ",dP.typeDimRed,"encoder saved in:", dP.model_pca,"\n")
    with open(dP.model_pca,'wb') as f:
        pickle.dump(spca, f)
        
    return A_encoded
        
#********************************************************************************
# Convert matrix data to saved scaled/PCA
#********************************************************************************
def runPCAValid(A, dP):
    import numpy as np
    from sklearn import preprocessing, decomposition
    
    with open(dP.model_pca,'rb') as f:
        spca = pickle.load(f)
    
    if dP.rescaleForPCA:
        with open(dP.model_scaling,'rb') as f:
            scaler = pickle.load(f)
        A_enc = spca.transform(scaler.transform(A))
    else:
        A_enc = spca.transform(A)
        
    return A_enc

#********************************************************************************
# Carry out statistics/plots for PCA analysis
#********************************************************************************
def statsPCA(En, A_r, Cl, dP):
    showDimRedplots = True
    
    with open(dP.model_pca,'rb') as f:
        pca = pickle.load(f)
    
    for i in range(0,pca.components_.shape[0]):
        print(' Score PC ' + str(i) + ': ' + '{0:.0f}%'.format(pca.explained_variance_ratio_[i] * 100))

    if showDimRedplots:
        import matplotlib.pyplot as plt
        from matplotlib import cm
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
        
#************************************
# Autoencoder
#************************************
def preAutoencoder(learnFile, validFile, dP):
    En, A, Cl, _ = readLearnFile(learnFile, dP)
    A_encoded = runAutoencoder(A, dP, Cl)
    
def runAutoencoder(A, dP, Cl):
    if checkTFVersion("2.16.0"):
        import tensorflow.keras as keras
    else:
        if dP.kerasVersion == 2:
            import tf_keras as keras
        else:
            import keras
    
    showDimRedplots = False

    m = keras.Sequential()
    m.add(keras.Input((A.shape[1],),sparse=True))
    m.add(keras.layers.Dense(A.shape[1]-1, activation='elu'))
    
    for i in range(A.shape[1]-1,2,-1):
        m.add(keras.layers.Dense(i-1,  activation='elu'))
    
    m.add(keras.layers.Dense(1,    activation='linear', name="bottleneck"))
    
    for i in range(2,A.shape[1],1):
        m.add(keras.layers.Dense(i,  activation='elu'))
    
    m.add(keras.layers.Dense(A.shape[1], activation='sigmoid'))

    print("  Training Autoencoder... \n")
    m.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam())
    history = m.fit(A, A, batch_size=dP.batch_size, epochs=dP.epochs, verbose=1)
    
    print("\n  Setting up Autoencoder input tensor... \n")
    
    encoder = keras.Model(inputs = m.inputs[0], outputs=m.get_layer('bottleneck').output)
    keras.Input((A.shape[1],))
    Zenc = encoder.predict(A)  # bottleneck representation
    Renc = m.predict(A)        # reconstruction
    
    saved_model_autoenc = os.path.splitext(dP.model_pca)[0]+".keras"
    print("\n  Autoencoder saved in:", saved_model_autoenc,"\n")
    encoder.save(saved_model_autoenc)
    
    if showDimRedplots:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.subplot(121)
        plt.title('Autoencoder')
        plt.scatter(Zenc[:,0], Zenc[:,1], c=Cl[:], s=8, cmap='tab10')
        plt.gca().get_xaxis().set_ticklabels([])
        plt.gca().get_yaxis().set_ticklabels([])

        plt.subplot(122)
        plt.title('Autoencoder')
        plt.scatter(Zenc, Zenc, c=Cl[:], s=8, cmap='tab10')
        plt.gca().get_xaxis().set_ticklabels([])
        plt.gca().get_yaxis().set_ticklabels([])

        plt.tight_layout()
        plt.show()
    
    return Zenc
    
#************************************
# Autoencoder - Alternative
#************************************
def runAutoencoder2(learnFile, testFile, dP):
    import tensorflow as tf
    if checkTFVersion("2.16.0"):
        import tensorflow.keras as keras
    else:
        if dP.kerasVersion == 2:
            import tf_keras as keras
        else:
            import keras
    
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
            
    
    En, A, Cl, _ = readLearnFile(learnFile, dP)
            
    shape = A.shape[1:]
    autoencoder = Autoencoder(dP.numDimRedComp, shape)
    autoencoder.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    
    if testFile is None:
        autoencoder.fit(A, A,
                epochs = dP.epochs,
                batch_size = dP.batch_size,
                shuffle=True,
                #validation_data=(x_test, x_test),
                validation_split=dP.cv_split
                )
    else:
        En_test, A_test, Cl_test, _ = readLearnFile(testFile, dP)
        autoencoder.fit(A, A,
                epochs = dP.epochs,
                batch_size = dP.batch_size,
                shuffle=True,
                validation_data=(A_test, A_test),
                )
                
        A_test_encoded = autoencoder.encoder(A_test).numpy()
        A_test_decoded = autoencoder.decoder(A_test_encoded).numpy()
        
    saved_model_autoenc = os.path.splitext(dP.model_pca)[0]+".keras"
    print("\n  Autoencoder saved in:", saved_model_autoenc,"\n")
    autoencoder.save(saved_model_autoenc)
    
    #print(autoencoder.encoder(A).numpy())
    return autoencoder.encoder(A).numpy()
    
#************************************
# Format subset
#************************************
def formatSubset(A, Cl, percent):
    import numpy as np
    from sklearn.model_selection import train_test_split
    A_train, A_cv, Cl_train, Cl_cv = \
    train_test_split(A, Cl, test_size=percent, random_state=None)
    uniCl = np.unique(Cl_cv).astype(int)
    if Cl_cv.shape[0] - uniCl.shape[0] > 0:
        print("  Classes with multiple data present.")
        print("\n  Unique classes in learning/validation set and corresponding number of members:\n")
        uni = np.ones(np.unique(Cl_cv).shape)
        for x in enumerate(uniCl):
            uni[x[0]] = np.count_nonzero(Cl_cv==uniCl[x[0]])
            if uni[x[0]] == 1:
                print(" {0:.0f}: {1:.0f} ".format(x[0],uni[x[0]]))
            else:
                print(" \033[1m {0:.0f}: {1:.0f} \033[0m".format(x[0],uni[x[0]]))
        flag = True
    else:
        print("\n  Unique classes in learning/validation set:")
        with np.printoptions(threshold=np.inf):
            print("  ",np.unique(Cl_cv).astype(int),"\n")
        flag = False
    return A_train, Cl_train, A_cv, Cl_cv, flag
    
#************************************
# Save Summary of predictions as CSV
#************************************
def saveSummaryFile(summaryFile, dP):
    import pandas as pd
    dframe = pd.DataFrame(summaryFile)
    dframe.to_csv(dP.summaryFileName, index=False, header=False)
    print("  Prediction summary saved in:",dP.summaryFileName,"\n")
    

#************************************
# Print NN Info
#************************************
def printParam(dP):
    print('\n  ================================================')
    print('  \033[1m ML\033[0m - Parameters')
    print('  ================================================')
    print('  Optimizer:','Adam',
                '\n  Hidden layers:', dP.HL,
                '\n  Activation function:','relu',
                '\n  L2:',dP.l2,
                '\n  Dropout:', dP.drop,
                '\n  Learning rate:', dP.l_rate,
                '\n  Learning decay rate:', dP.l_rdecay)
    if dP.fullSizeBatch:
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
# Print NN Info
#************************************
def printParamDF(dP):
    print('\n  ================================================')
    print('  \033[1m',dP.typeDF,dP.mode,' \033[0m- Parameters')
    print('  ================================================')
    print('   Number of estimators:',dP.n_estimators,
                '\n   Max depth:', dP.max_depth,
                '\n   Max features:',dP.max_features,
                '\n   Epochs/Max number of iterations:',dP.epochs,
                '\n   Cross validation split:', str(dP.cv_split*100)+'%')
    if dP.fullSizeBatch:
        print('   Batch size: full')
    else:
        print('   Batch size:', dP.batch_size)
    print('   Train on full dataset:',dP.trainFullData)
    print('   Number of labels:', dP.numLabels)
    print('   Random state:', dP.random_state)
    print('   Normalize:', dP.normalize)
    if dP.runDimRedFlag:
        print('   Dimensionality reduction algorithm:', dP.typeDimRed)
        print('   Number of dimensionality reduction components:', dP.numDimRedComp)
    
    #print('  ================================================\n')

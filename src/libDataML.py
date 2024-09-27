# -*- coding: utf-8 -*-
'''
***********************************************************
* libDataML - Library for DataML
* v2024.09.26.2
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
                if os.path.isfile(dP.model_name) is False:
                    model_name = os.path.splitext(dP.model_name)[0]+".h5"
                else:
                    model_name = dP.model_name
                model = keras.models.load_model(model_name)
                #model = keras.saving.load_model(model_name)
            else:
                model_name = dP.model_name
                model = keras.models.Sequential()
                model.add(keras.layers.TFSMLayer(model_name, call_endpoint='serve'))
            print("  Model name:",model_name)
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
    if checkTFVersion("2.16.0"):
        import tensorflow.keras as keras
        model = keras.models.load_model(dP.model_name)
    else:
        if dP.kerasVersion == 2:
            import tf_keras as keras
            model = keras.models.load_model(dP.model_name)
        else:
            import keras
            model = keras.layers.TFSMLayer(os.path.splitext(dP.model_name)[0], call_endpoint='serve')
            
    converter = tf.lite.TFLiteConverter.from_keras_model(model)    # TF2.3 and higher only for full EdgeTPU support.

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
    from packaging import version    
    if dP.useTFlitePred:
        print("\n TensorFlow (Lite) v.",tf.version.VERSION,"\n")
    else:
        print("\n TensorFlow v.",tf.version.VERSION,"\n" )
        
def checkTFVersion(vers):
    import tensorflow as tf
    from packaging import version
    v = version.parse(tf.version.VERSION)
    return v < version.parse(vers)


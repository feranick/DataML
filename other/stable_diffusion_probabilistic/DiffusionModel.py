#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* DiffusionModel
* Data Augmentation via Diffusion Model
* version: v2024.12.20.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py, pickle, configparser
from libDataML import *
import tensorflow as tf
import keras

#***************************************************
# This is needed for installation through pip
#***************************************************
def DiffusionModel():
    main()

#************************************
# Parameters definition
#************************************
class Conf():
    def __init__(self):
        self.appName = "DiffusionModel"
        confFileName = "DiffusionModel.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "./"
        
        self.norm_file = self.model_directory+"norm_file.pkl"
        self.numLabels = 1
        
        self.betas = np.linspace(self.beta_start, self.beta_end, self.time_steps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        
        # Non-linear time - original
        alphas_cumprod = np.cumprod(self.alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.one_minus_sqrt_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        '''
        # Linear time - testing
        self.sqrt_alphas_cumprod = self.alphas
        self.one_minus_sqrt_alphas_cumprod = self.betas
        '''
    def diffModDef(self):
        self.conf['Parameters'] = {
            'saveAsTxt' : True,
            'deepModel' : True,
            'normalNoise' : True,
            'reinforce' : True,
            'encoded_dim' : 1,
            'batch_size' : 16,
            'epochs' : 200,
            'time_steps' : 50,
            'beta_start' : 1e-4,
            'beta_end' : 0.02,
            'l_rate' : 0.1,
            'l_rdecay' : 0.01,
            'l2' : 1e-5,
            'stdNoiseFactor' : 1,
            'numAdditions' : 300,
            'excludeZeroFeatures' : True,
            'removeSpurious' : True,
            'normalize' : False,
            'normalizeLabel' : False,
            }
            
    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.diffModDef = self.conf['Parameters']
        
            self.saveAsTxt = self.conf.getboolean('Parameters','saveAsTxt')
            self.deepModel = self.conf.getboolean('Parameters','deepModel')
            self.reinforce = self.conf.getboolean('Parameters','reinforce')
            self.normalNoise = self.conf.getboolean('Parameters','normalNoise')
            self.batch_size = self.conf.getint('Parameters','batch_size')
            self.encoded_dim = self.conf.getint('Parameters','encoded_dim')
            self.epochs = self.conf.getint('Parameters','epochs')
            self.time_steps = self.conf.getint('Parameters','time_steps')
            self.beta_start = self.conf.getfloat('Parameters','beta_start')
            self.beta_end = self.conf.getfloat('Parameters','beta_end')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.l_rdecay = self.conf.getfloat('Parameters','l_rdecay')
            self.l2 = self.conf.getfloat('Parameters','l2')
            self.stdNoiseFactor = self.conf.getfloat('Parameters','stdNoiseFactor')
            self.numAdditions = self.conf.getint('Parameters','numAdditions')
            self.excludeZeroFeatures = self.conf.getboolean('Parameters','excludeZeroFeatures')
            self.removeSpurious = self.conf.getboolean('Parameters','removeSpurious')
            self.normalize = self.conf.getboolean('Parameters','normalize')
            self.normalizeLabel = self.conf.getboolean('Parameters','normalizeLabel')
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
    
    '''
    import random
    random.randint(1, 2000)
    seed_value = random.randint(1, 2000)
    tf.random.set_seed(seed_value)
    '''

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 DiffusionModel.py <learnData>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    dP = Conf()

    En, A, M = readLearnFile(dP, sys.argv[1], True)
    
    if dP.normalize:
        with open(dP.norm_file, "rb") as f:
            norm = pickle.load(f)
        newA = norm.transform_inverse(M[1:,:])
    else:
        newA = A
        norm = 0
            
    # Load model or create a new one.
    saved_diff_model = dP.model_directory + os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_diffModel.keras"

    if dP.reinforce and os.path.exists(saved_diff_model):
        print(" Loading Diffuse Model:",saved_diff_model,"\n")
        model = keras.saving.load_model(saved_diff_model, custom_objects={'DiffusionModel': DiffusionModel})
    else:
        print(" Initializing new Diffuse Model\n")
        model = DiffusionModel(feature_dim=A.shape[1], time_embedding_dim=A.shape[1], encoded_dim = dP.encoded_dim, conf = dP)
            
    trained_model = train_diffusion_model(model, A, saved_diff_model, dP)
    A_tmp = sample_from_model(trained_model, A, num_samples=dP.numAdditions, feature_dim=A.shape[1], conf=dP).numpy()
    
    if dP.removeSpurious:
        A_tmp, numAddedData = removeSpurious(A, A_tmp, norm, dP)
        #newA = removeSpurious(A, newA, norm)
        print("\n  Spurious data removed.")
        tag = '_noSpur'
    else:
        tag = ''
        numAddedData = dP.numAdditions
    
    newA = np.vstack([newA, A_tmp])
    newTrain = np.vstack([En, newA])
    print("\n  Added",str(numAddedData),"new data")
    
    np.set_printoptions(suppress=True)
    
    newFile = dP.model_directory + os.path.splitext(os.path.basename(sys.argv[1]))[0] + '_diffModNumAdd' + str(numAddedData) + tag
    saveLearnFile(dP, newA, newFile, "")
    
    file_csv = os.path.splitext(os.path.basename(newFile))[0]+".csv"
    np.savetxt(file_csv, newTrain, delimiter=",", fmt='%.2f')
    print("  Saving matrix for correlation analysis\n  ",file_csv,"\n")
    
    
#*******************************************
# Helper function to create time embeddings
#*******************************************
def get_time_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    exponent = tf.math.log(10000.0) / (half_dim - 1)
    embeddings = tf.range(half_dim, dtype=tf.float32) * -exponent
    embeddings = tf.exp(embeddings)
    embeddings = tf.cast(timesteps, tf.float32)[:, None] * embeddings[None, :]
    embeddings = tf.concat([tf.sin(embeddings), tf.cos(embeddings)], axis=-1)
    return embeddings

#*******************************************
# Define the model
#*******************************************
@keras.saving.register_keras_serializable()
class DiffusionModel(keras.Model):
    def __init__(self, feature_dim, time_embedding_dim, encoded_dim, conf, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.time_embedding_dim = time_embedding_dim
        self.encoded_dim = encoded_dim
        self.conf = conf
        
        self.time_embedding_layer = keras.layers.Dense(feature_dim, activation="relu")
        if self.conf.deepModel and feature_dim > encoded_dim+2:
            self.dense1=[]
            self.dense2=[]
            for i in range(feature_dim-1,encoded_dim+1,-1):
                self.dense1.append(keras.layers.Dense(i-1, activation="relu", kernel_regularizer=keras.regularizers.l2(conf.l2)))
            for i in range(encoded_dim+2,feature_dim,1):
                self.dense2.append(keras.layers.Dense(i, activation="relu",kernel_regularizer=keras.regularizers.l2(conf.l2)))
            print(" Using Deep Diffusion Model, number of layers:",str(len(self.dense1)+len(self.dense2)+2),"\n")
        else:
            self.dense1 = keras.layers.Dense(encoded_dim, activation="relu", kernel_regularizer=keras.regularizers.l2(conf.l2))
            print(" Using single layer diffusion model")
        self.output_layer = keras.layers.Dense(feature_dim, activation=None)
                
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "trainable": self.trainable,
                "feature_dim": self.feature_dim,
                "time_embedding_dim": self.time_embedding_dim,
                "encoded_dim": self.encoded_dim,
            }
        )
        return config
       
    def build(self, input_shape):
        # Ensure input shapes are properly built
        self.built = True

    def call(self, inputs, training=False):
        x, t = inputs
        time_embedding = get_time_embedding(t, self.time_embedding_dim)
        time_features = self.time_embedding_layer(time_embedding)
        
        x = x + time_features
        if self.conf.deepModel and self.feature_dim > self.encoded_dim+2:
            for i in range(len(self.dense1)):
                x = self.dense1[i](x)
            for i in range(len(self.dense2)):
                x = self.dense2[i](x)
        else:
            x = self.dense1(x)
        return self.output_layer(x)
    
#*******************************************
# Loss function
#*******************************************
def diffusion_loss(model, x_start, t, noise, dP):
    # Gather the scaling factors for the current timestep
    alpha_t = tf.gather(dP.sqrt_alphas_cumprod, t)
    one_minus_alpha_t = tf.gather(dP.one_minus_sqrt_alphas_cumprod, t)

    # Reshape scaling factors to broadcast properly
    alpha_t = tf.expand_dims(alpha_t, axis=-1)  # Shape: [BATCH_SIZE, 1]
    one_minus_alpha_t = tf.expand_dims(one_minus_alpha_t, axis=-1)  # Shape: [BATCH_SIZE, 1]
    
    # Compute the noisy input
    noisy_input = alpha_t * x_start + one_minus_alpha_t * noise
    #noisy_input = one_minus_alpha_t * x_start + alpha_t * noise
    
    #print((alpha_t*x_start).numpy(),(one_minus_alpha_t*noise).numpy(), noisy_input.numpy())

    # Predict the noise
    predicted_noise = model([noisy_input, t], training=True)

    # Compute the loss
    #return tf.reduce_mean(tf.square(noise - predicted_noise))
    return tf.reduce_mean(tf.abs(noise - predicted_noise))
    
#*******************************************
# Train loop
#*******************************************
def train_diffusion_model(model, data, file, dP):
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=dP.l_rate,
            decay_steps=dP.epochs,
            decay_rate=dP.l_rdecay)
    #optimizer = keras.optimizers.Adam(learning_rate=dP.l_rate)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    for t in range(0,dP.time_steps,):
        #noise = np.abs(tf.random.normal(shape=data.shape))
        if dP.normalNoise:
            noise = random_normal(data, data.shape[0], data.shape[1], dP)
        else:
            noise = random_uniform(data, data.shape[0], data.shape[1], dP)
                
        alpha_t = tf.gather(dP.sqrt_alphas_cumprod, t)
        one_minus_alpha_t = tf.gather(dP.one_minus_sqrt_alphas_cumprod, t)
        
        #print("alpha_t:", alpha_t.numpy(),one_minus_alpha_t.numpy())
        
        noisy_input = alpha_t * data + one_minus_alpha_t * noise
                
    for epoch in range(dP.epochs):
        np.random.shuffle(data)
        for i in range(0, len(data), dP.batch_size):
            batch = data[i:i + dP.batch_size]
            batch = tf.convert_to_tensor(batch, dtype=tf.float32)
                        
            t = tf.random.uniform((len(batch),), minval=0, maxval=dP.time_steps, dtype=tf.int32)
            #noise = tf.random.normal(shape=batch.shape)
            if dP.normalNoise:
                noise = random_normal(data, batch.shape[0], data.shape[1], dP)
            else:
                noise = random_uniform(data, batch.shape[0], data.shape[1], dP)

            with tf.GradientTape() as tape:
                loss = diffusion_loss(model, batch, t, noise, dP)
                
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        print(f"Epoch {epoch + 1}/{dP.epochs}, Loss: {loss.numpy()}")
    
    print("\n  Diffusion model saved in:", file,"\n")
    model.save(file)

    return model
    
#*******************************************
# Sampling function
#*******************************************
def sample_from_model(model, A, num_samples, feature_dim, conf):
    #x = np.abs(tf.random.normal(shape=(num_samples, feature_dim)))
    #x = tf.random.uniform(shape=(num_samples, feature_dim), minval = 0, maxval = 1)
    x = random_normal(A, num_samples, feature_dim, conf)
    #x = random_uniform(A, num_samples, feature_dim, conf)
        
    for t in reversed(range(conf.time_steps)):
        t_tensor = tf.fill((num_samples,), t)
        predicted_noise = model([x, t_tensor], training=False)
        alpha_t = tf.gather(conf.sqrt_alphas_cumprod, t)
        one_minus_alpha_t = tf.gather(conf.one_minus_sqrt_alphas_cumprod, t)
        beta_t = tf.gather(conf.betas, t)
        x = (x - beta_t * predicted_noise) / alpha_t
        
    return x

# custom random normal generator
def random_normal(A, num_samples, feature_dim, conf):
    randA = np.empty(shape=(0, num_samples))
    
    for i in range(feature_dim):
        #print(np.mean(A[:,i]), np.std(A[:,i]))
        a = []
        for j in range(num_samples):
            tmp = np.abs(np.random.normal(loc = np.mean(A[:,i]), scale = np.std(A[:,i])*conf.stdNoiseFactor))
            a.append(tmp)
        randA = np.vstack([randA, a])
    return randA.T
  
# custom random uniform generator
def random_uniform(A, num_samples, feature_dim, conf):
    randA = np.empty(shape=(0, num_samples))
    
    def getMin(A):
        return np.min(A[np.where(A != 0)])
    
    for i in range(feature_dim):
        #print(getMin(A[:,i]), np.max(A[:,i]))
        a = []
        for j in range(num_samples):
            tmp = np.random.uniform(low = getMin(A[:,i]), high = np.max(A[:,i]))
            a.append(tmp)
        randA = np.vstack([randA, a])
    return randA.T
    
#*******************************************
# Create from single data
#*******************************************
def single_from_model(model, x, conf):
    #x = tf.random.normal(shape=(num_samples, feature_dim))
    #print(x)
    
    for t in reversed(range(conf.time_steps)):
        t_tensor = tf.fill((num_samples,), t)
        predicted_noise = model([x, t_tensor], training=False)
        
        alpha_t = tf.gather(conf.sqrt_alphas_cumprod, t)
        one_minus_alpha_t = tf.gather(conf.one_minus_sqrt_alphas_cumprod, t)
        beta_t = tf.gather(conf.betas, t)

        x = (x - beta_t * predicted_noise) / alpha_t
    return x

#************************************
# Generate data from Autoencoder
#************************************
def getAmin(A):
    A_min = []
    for i in range(A.shape[1]):
        A_min_single = min(x for x in A[:,i] if x != 0)
        A_min = np.hstack([A_min,A_min_single])
    A_min = np.asarray(A_min)
    return A_min

def getAmax(A):
    A_max = []
    for i in range(A.shape[1]):
        A_max_single = max(x for x in A[:,i] if x != 0)
        A_max = np.hstack([A_max,A_max_single])
    A_max = np.asarray(A_max)
    return A_max
    
def removeSpurious(A, T, norm, dP):
    if dP.normalize:
        A_min = norm.transform_inverse(np.asarray([getAmin(A)]))[0]
        A_max = norm.transform_inverse(np.asarray([getAmax(A)]))[0]
    else:
        A_min = getAmin(A)
        A_max = getAmax(A)
    
    '''
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i,j] < A_min[j]:
                T[i,j] = A_min[j]
            if T[i,j] > A_max[j]:
                T[i,j] = A_max[j]
            #if T[i,j] < A_min[j] or T[i,j] > A_max[j]:
            #    T[i,j] = 0
    '''
    for i in range(T.shape[0]):
        flag = False
        for j in range(T.shape[1]):
            if T[i,j] < A_min[j] or T[i,j] > A_max[j]:
                flag = True
        if flag:
            T[i] = 0
    
    
    # Remove rows with all zero values
    T = T[np.any(T != 0, axis=1)]
    #T = T[np.all(T != 0, axis=1)]
    
    print(T)
    
    return T, T.shape[0]

#************************************
# Open Learning Data
#************************************
def readLearnFile(dP, learnFile, newNorm):
    print(" Opening learning file:\n  "+learnFile+"\n")
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
        print("\033[1m" + " Learning file not found \n" + "\033[0m")
        return

    if dP.normalize:
        print("  Normalization of feature matrix to 1")
        if newNorm:
            print("  Normalization parameters saved in:", dP.norm_file,"\n")
            norm = Normalizer(M, dP)
        else:
            print("  Normalization parameters from:", dP.norm_file,"\n")
            with open(dP.norm_file, "rb") as f:
                norm = pickle.load(f)
        M = norm.transform(M)
        norm.save()

    En = M[0,:]
    A = M[1:,:]
    Cl = M[1:,0]
    
    return En, A, M

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

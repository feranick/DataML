#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
* Convert TF models (TF, Keras) into TF.Lite
* 20191015a
* Uses: TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)
import tensorflow as tf
import sys, os.path, h5py

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3.py ConvertToTFLite <Model in HDF5 format>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        print(' Converting',sys.argv[1],'to TF.Lite...\n')
        convertModelToTFLite(sys.argv[1])

#************************************
# Convert TF Model to TF.Lite
#************************************
def convertModelToTFLite(model_file):
    convFile = os.path.splitext(model_file)[0]+'.tflite'
    try:
        model = tf.keras.models.load_model(model_file)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)    # TensorFlow 2.x
    except:
        converter = tf.lite.TFLiteConverter.from_keras_model_file(model_file)  # TensorFlow 1.x
        
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    open(convFile, "wb").write(tflite_model)
    print(' Converted model saved to:',convFile,'\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************************
* Convert TF Keras V3 models into TF.Lite
* v2024.02.28.1
* Uses: TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)
import sys, os.path, h5py

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3.py ConvertToTFLite <Model in Keras format>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        print(' Converting',sys.argv[1],'to TF.Lite...\n')
        convertModelToTFLite(sys.argv[1])

#************************************
# Convert TF Model to TF.Lite
#************************************
def convertModelToTFLite(model_file):
    import tensorflow as tf
    import keras
    
    convFile = os.path.splitext(model_file)[0]+'.tflite'
    try:
        model = keras.layers.TFSMLayer(model_file, call_endpoint='serve') # TensorFlow >= 2.16.0
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    except RuntimeError as err:
        print(err);
        
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)
    open(convFile, "wb").write(tflite_model)
    print(' Converted model saved to:',convFile,'\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())



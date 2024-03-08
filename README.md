# DataML
DataML Regression and Classification of sparse data.
- Currently supported ML architectures:
   - DNNClassifier (TensorFlow, TensorFlow-Lite)
- Required libraries for prediction:
   - tensorflow (version >= 2.13.x, >= 2.16.1 recommended)
   - Optional: tensorflow-lite (v >= 2.13, >= 2.16.1 recommended)
   - Optional: [tensorflow-lite runtime](https://www.tensorflow.org/lite/guide/python) 
   - Optional: tensorflow-lite runtime with [Coral EdgeTPU](https://coral.ai/docs/accelerator/get-started/)

Installation
=============
## Installation from available wheel package
If available from the main site, you can install SpectraKeras by running:

    python3 -m pip install --upgrade spectrakeras-20231215a0-py3-none-any.whl
    
SpectraKeras_CNN and Spectrakeras_MLP are available directly from the command line.
NOTE: The Utilities in the `Utilities` folder are not included in the package, and can be run locally as needed.

## Make your own wheel package
Make sure you have the PyPA build package installed:

    python3 -m pip install --upgrade build
    
To build the wheel package rom the `SpectraKeras` folder run:

    python3 -m build
    
A wheel package is available in the subfolder `dir`. You can install it following the instructions shown above.

## Compatibility and dependences
This software requires Python (3.8 or higher). It has been tested with Python 3.6 or higher which is the recommended platform. It is not compatible with python 2.x. Additional required packages:

    numpy
    scikit-learn (>=0.18)
    scipy
    matplotlib
    pandas
    pydot
    graphviz
    h5py
    tensorflow (>= 2.13.x)
    
In addition, these packages may be needed depending on your platform (via ```apt-get``` in debian/ubuntu or ```port``` in OSX):
    
    python3-tk
    graphviz

These are found in Unix based systems using common repositories (apt-get for Debian/Ubuntu Linux, or MacPorts for MacOS). More details in the [scikit-learn installation page](http://scikit-learn.org/stable/install.html).

[TensorFlow](https://www.tensorflow.org) is needed only if flag is activated. Instructions for Linux and MacOS can be found in [TensorFlow installation page](https://www.tensorflow.org/install/). Pip installation is the easiest way to get going. Tested with TensorFlow v.1.15+. TensorFlow 2.x (2.3 or higher preferred) is the currently sipported release. 

Prediction can be carried out using the regular tensorflow, or using [tensorflow-lite](https://www.tensorflow.org/lite/) for [quantized models](https://www.tensorflow.org/lite/performance/post_training_quantization). Loading times of tflite (direct or via [tflite-runtime](https://www.tensorflow.org/lite/guide/python)) are significantly faster than tensorflow with minimal loss in accuracy. SpectraKeras provides an option to convert tensorflow models to quantized tflite models. TFlite models have been tested in Linux x86-64, arm7 (including Raspberry Pi3) and aarm64, MacOS, Windows. For using quantized model (specifically when deployed on Coral EdgeTPU), TF 2.13 or higher is recommended. 

Usage
===================
Train (Random cross validation):
  `python3 DataML.py -t <learningFile>`

 Train (with external validation):
  `python3 DataML.py -t <learningFile> <validationFile> `

 Train (with external validation, with labels normalized with pkl file):
  `python3 DataML.py -t <learningFile> <validationFile> <pkl normalization file>`

 Predict (no label normalization used):
  `python3 DataML.py -p <testFile>`

 Predict (labels normalized with pkl file):
  `python3 DataML.py -p <testFile> <pkl normalization file>`

 Batch predict (no label normalization used):
  `python3 DataML.py -b <folder>`

 Batch predict (labels normalized with pkl file):
  `python3 DataML.py -b <folder> <pkl normalization file>`

 Batch predict on validation data in single file (no label normalization used):
  `python3 DataML.py -v <singleValidationFile>`

 Batch predict on validation data in single file (labels normalized with pkl file):
  `python3 DataML.py -v <singleValidationFile> <pkl normalization file>`


Formatting input file for training
========================
The main idea behind the software is to train classification or regression models from raw data. So, suppose one has training files similar to this, where the first column is the a progressive index the second is the value of the quantity for a specific parameter:

```
1  123
2  140
3  180
4  150
...
```

Let's say this file correspond to label 1, and now one has a collection of files that will be used for training each with its own label, the input file will look like this:

```
0  1  2  3  4 ...
lab1  123 140  180  150  ...
lab2 ... ... ... ... ...
```
Essentially each line in the input file corresponds to a training file with its label. during training the model will learn (either through a simple deep MLP network using `DataML.py`, to extract features needed for prediction. 

Of course it is not expected that the user manually compiles the training file. For that,based on a CSV file with the raw data, [`MasterDatamaker.py`](https://github.com/feranick/DataML/tree/master/src/utilities/MasterDataMaker.py) is available in the [`Utilities`](https://github.com/feranick/DataML/tree/master/src/utilities) folder, that can be used to automatically create such files. Basically you can run from the folder where you have your spectra:

`python3 DataMaker.py <paramFile> <pred column> `

Specific DataMakers for specific applications may be available depending on the project. 

One can use the same to create a validation file, or you can use [other scripts](https://github.com/feranick/DataML/tree/master/src/utilities) also provided to split the training set into training+validation. That can be done randomly within SpectraKeras, but the split will be different every time you run it.

Once models are trained trained, prediction on individual files can be made using simply formatted ASCII files (like in the example above).

from keras import models
import numpy as np

# ====================================================================================
#                                   INFORMATION 
# ====================================================================================
# (FIBER -> COMPOSITE)
# This code takes fiber properties and predicts composite properties.
# The prediction is performed using a suite of neural networks.
# Date Created: 7/14/2021

# ====================================================================================
#                                   USER INPUTS 
# ====================================================================================
E11_fiber = 487.3 # Young's modulus of the fiber E11 (range: 80-500) [GPa]
E22_E33_fiber = 19.8 # Young's modulus of the fiber E22=E22 (range: 2-20) [GPa]
G12_G13_fiber = 12.8 # Shear modulus of the fiber G12=G13 (range: 1-15) [GPa]
v12_v13_fiber = 0.33 # Poisson's ratio of the fiber v12=v13 (range: 0.2-0.4)
v23_fiber = 0.63 # Poisson's ratio of the fiber v23 (range: 0.3-0.7)
FVF = 0.43 # Fiber volume fraction (range: 0.3-0.75)
E_matrix = 3.3 # Young's modulus of the matrix (range: 1-4) [GPa]
v_matrix = 0.43 # Poisson's ratio of the matrix (range: 0.3-0.45)

# ====================================================================================
#                               LOAD TRAINED MODELS 
# ====================================================================================
# Load trained composite property NN
model_C_E11 = models.load_model('Comp_Prop_E11.h5')
model_C_E22_E33 = models.load_model('Comp_Prop_E22_E33.h5')
model_C_G12_G13 = models.load_model('Comp_Prop_G12_G13.h5')
model_C_v12_v13 = models.load_model('Comp_Prop_v12_v13.h5')
model_C_v23 = models.load_model('Comp_Prop_v23.h5')

# Load normalization parameters for composite properties
Xmax_C = np.array([0.75,500.0,20.0,15.0,0.4,0.7,4.0,0.45])
Xmin_C = np.array([0.3,80.0,2.0,1.0,0.2,0.3,1.0,0.3])
dmax_C_E11, dmin_C_E11 = [370.37097,24.676718]
dmax_C_E22_E33, dmin_C_E22_E33 = [13.923995,1.2882799]
dmax_C_G13_G12, dmin_C_G13_G12 = [6.4756,0.4628]
dmax_C_v12_v13, dmin_C_v12_v13 = [0.44033787,0.21886805]
dmax_C_v23, dmin_C_v23 = [0.78672594,0.3209443]

# ====================================================================================
#              PREDICT PROPERTIES (FIBER -> COMPOSITE) 
# ====================================================================================
# Generate input vector for composite NN
X_raw = np.array([[FVF,E11_fiber,E22_E33_fiber,G12_G13_fiber,v12_v13_fiber,v23_fiber,E_matrix,v_matrix]])

# Normalize composite data
X_C = (X_raw-Xmin_C)/(Xmax_C-Xmin_C)

# Predict COMPOSITE properties
PredictedLabels_C_E11 = model_C_E11.predict(X_C)
ActualPredictedLabels_C_E11 = PredictedLabels_C_E11*(dmax_C_E11-dmin_C_E11)+dmin_C_E11

PredictedLabels_C_E22_E33 = model_C_E22_E33.predict(X_C)
ActualPredictedLabels_C_E22_E33 = PredictedLabels_C_E22_E33*(dmax_C_E22_E33-dmin_C_E22_E33)+dmin_C_E22_E33

PredictedLabels_C_G12_G13 = model_C_G12_G13.predict(X_C)
ActualPredictedLabels_C_G12_G13 = PredictedLabels_C_G12_G13*(dmax_C_G13_G12-dmin_C_G13_G12)+dmin_C_G13_G12

PredictedLabels_C_v12_v13 = model_C_v12_v13.predict(X_C)
ActualPredictedLabels_C_v12_v13 = PredictedLabels_C_v12_v13*(dmax_C_v12_v13-dmin_C_v12_v13)+dmin_C_v12_v13

PredictedLabels_C_v23 = model_C_v23.predict(X_C)
ActualPredictedLabels_C_v23 = PredictedLabels_C_v23*(dmax_C_v23-dmin_C_v23)+dmin_C_v23

# print COMPOSITE properties
print('COMPOSITE E11 [GPa]:',ActualPredictedLabels_C_E11[0][0])
print('COMPOSITE E22=E33 [GPa]:',ActualPredictedLabels_C_E22_E33[0][0])
print('COMPOSITE G12=G13 [GPa]:',ActualPredictedLabels_C_G12_G13[0][0])
print('COMPOSITE v12=v13:',ActualPredictedLabels_C_v12_v13[0][0])
print('COMPOSITE v23:',ActualPredictedLabels_C_v23[0][0])



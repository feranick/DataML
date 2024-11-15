from keras import models
import numpy as np

# ====================================================================================
#                                   INFORMATION 
# ====================================================================================
# (COMPOSITE -> FIBER)
# This code predicts fiber properties given composite and matrix properties.
# The prediction is performed using a suite of neural networks.
# Date Created: 7/14/2021

# ====================================================================================
#                                   USER INPUTS 
# ====================================================================================
E11_composite = 210.8 # Young's modulus of the composite E11 (range: 25-370) [GPa]
E22_E33_composite = 7.1 # Youngs modulus of the composite E22=E33 (range: 1-14) [GPa]
G12_G13_composite = 2.4 # Shear modulus of the composite G12=G13 (range: 0.5-6.5) [GPa]
v12_v13_composite = 0.39 # Poisson's ratio of the composite v12=v13 (range: 0.2-0.4) 
v23_composite = 0.7 # Poisson's ratio of the composite v23 (range: 0.3-0.7)
FVF = 0.43 # Fiber volume fraction (range: 0.3-0.75)
E_matrix = 3.3 # Young's modulus of the matrix (range: 1-4) [GPa]
v_matrix = 0.43 # Poisson's ratio of the matrix (range: 0.3-0.45)

# ====================================================================================
#                               LOAD TRAINED MODELS 
# ====================================================================================
# Load trained reverse fiber property NN
model_F_E11 = models.load_model('Fiber_Prop_E11.h5')
model_F_E22_E33 = models.load_model('Fiber_Prop_E22_E33.h5')
model_F_G12_G13 = models.load_model('Fiber_Prop_G13_G12.h5')
model_F_v12_v13 = models.load_model('Fiber_Prop_v12_v13')
model_F_v23 = models.load_model('Fiber_Prop_v23')

# Load normalization parameters for fiber properties
Xmax_F_E11, Xmin_F_E11, dmax_F_E11, dmin_F_E11 = np.array([4.0,0.45,371.7898,13.914589,6.4756,0.44033787,0.78672594,0.75]),np.array([ 1.0,0.3,24.676718,1.2882799,0.4628,0.21886805,0.3209443,0.3]),500.0,80.0
Xmax_F_E22_E33, Xmin_F_E22_E33, dmax_F_E22_E33, dmin_F_E22_E33 = np.array([4.0,0.45,371.7898,13.923995,6.4756,0.44033787,0.78672594,0.75]),np.array([1.0,0.3,24.676718,1.2882799,0.4628,0.21886805,0.3209443,0.3]),20.0,2.0
Xmax_F_G13_G12, Xmin_F_G13_G12, dmax_F_G13_G12, dmin_F_G13_G12 = np.array([4.0,0.45,371.7898,13.923995,6.4756,0.44033787,0.78672594,0.75]),np.array([1.0,0.3,24.676718,1.2882799,0.4628,0.21886805,0.3209443,0.3]),15.0,1.0
Xmax_F_v12_v13, Xmin_F_v12_v13, dmax_F_v12_v13, dmin_F_v12_v13 = np.array([4.0,0.45,370.37097,13.923995,6.4756,0.44033787,0.78672594,0.75]),np.array([1.0,0.3,24.676718,1.2882799,0.4628,0.21886805,0.3209443,0.3]),0.4,0.2
Xmax_F_v23, Xmin_F_v23, dmax_F_v23, dmin_F_v23 = np.array([4.0,0.45,371.7898,13.923995,6.4756,0.44033787,0.78672594,0.75,]),np.array([1.0,0.3,24.676718,1.2882799,0.4628,0.21886805,0.3209443,0.3]),0.7,0.3

# ====================================================================================
#              PREDICT PROPERTIES (COMPOSITE -> FIBER) 
# ====================================================================================
# Generate the input vector for reverse fiber NN
X_raw = np.array([[E_matrix,v_matrix,E11_composite,E22_E33_composite,G12_G13_composite,v12_v13_composite,v23_composite,FVF]])

# Normalize input composite data
X_F_E11 = (X_raw-Xmin_F_E11)/(Xmax_F_E11-Xmin_F_E11)
X_F_E22_E33 = (X_raw-Xmin_F_E22_E33)/(Xmax_F_E22_E33-Xmin_F_E22_E33)
X_F_G12_G13 = (X_raw-Xmin_F_G13_G12)/(Xmax_F_G13_G12-Xmin_F_G13_G12)
X_F_v12_v13 = (X_raw-Xmin_F_v12_v13)/(Xmax_F_v12_v13-Xmin_F_v12_v13)
X_F_v23 = (X_raw-Xmin_F_v23)/(Xmax_F_v23-Xmin_F_v23)

# Predict FIBER properties
PredictedLabels_F_E11 = model_F_E11.predict(X_F_E11)
ActualPredictedLabels_F_E11 = PredictedLabels_F_E11*(dmax_F_E11-dmin_F_E11)+dmin_F_E11

PredictedLabels_F_E22_E33 = model_F_E22_E33.predict(X_F_E22_E33)
ActualPredictedLabels_F_E22_E33 = PredictedLabels_F_E22_E33*(dmax_F_E22_E33-dmin_F_E22_E33)+dmin_F_E22_E33

PredictedLabels_F_G12_G13 = model_F_G12_G13.predict(X_F_G12_G13)
ActualPredictedLabels_F_G12_G13 = PredictedLabels_F_G12_G13*(dmax_F_G13_G12-dmin_F_G13_G12)+dmin_F_G13_G12

PredictedLabels_F_v12_v13 = model_F_v12_v13.predict(X_F_v12_v13)
ActualPredictedLabels_F_v12_v13 = PredictedLabels_F_v12_v13*(dmax_F_v12_v13-dmin_F_v12_v13)+dmin_F_v12_v13

PredictedLabels_F_v23 = model_F_v23.predict(X_F_v23)
ActualPredictedLabels_F_v23 = PredictedLabels_F_v23*(dmax_F_v23-dmin_F_v23)+dmin_F_v23

# Print FIBER properties
print('FIBER E11 [GPa]:',ActualPredictedLabels_F_E11[0][0])
print('FIBER E22=E33 [GPa]:',ActualPredictedLabels_F_E22_E33[0][0])
print('FIBER G12=G13 [GPa]:',ActualPredictedLabels_F_G12_G13[0][0])
print('FIBER v12=v13:',ActualPredictedLabels_F_v12_v13[0][0])
print('FIBER v23:',ActualPredictedLabels_F_v23[0][0])


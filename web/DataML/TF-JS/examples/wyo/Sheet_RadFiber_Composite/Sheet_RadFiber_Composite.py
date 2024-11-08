from keras import models
import numpy as np

# ====================================================================================
#                                   INFORMATION 
# ====================================================================================
# (WAVY SHEET -> RADIAL FIBER -> COMPOSITE)
# This code takes wavy sheet properties and predicts the final composite properties.
# The prediction is performed using a suite of neural networks.
# Date Created: 7/14/2021

# ====================================================================================
#                                   USER INPUTS 
# ====================================================================================
E11_sheet = 419.5058583 # Young's modulus of the sheet E11 (range: 80-500) [GPa]
G12p_sheet = 27.4306997 # Shear modulus of the sheet G12 (range: 1-300) [GPa]
FVF = 0.4 # Fiber volume fraction (range: 0.3-0.75)
E_matrix = 4.0 # Young's modulus of the matrix (range: 1-4) [GPa]
v_matrix = 0.3 # Poisson's ratio of the matrix (range: 0.3-0.45)

# ====================================================================================
#                               LOAD TRAINED MODELS 
# ====================================================================================
# Load trained radial fiber property NN
model_F_E11 = models.load_model('Rad_Fiber_Prop_E11.h5')
model_F_E22_E33 = models.load_model('Rad_Fiber_Prop_E22_E33.h5')
model_F_G12_G13 = models.load_model('Rad_Fiber_Prop_G12_G13.h5')
model_F_v12_v13 = models.load_model('Rad_Fiber_Prop_v12_v13.h5')
model_F_v23 = models.load_model('Rad_Fiber_Prop_v23.h5')

# Load trained composite property NN
model_C_E11 = models.load_model('Comp_Prop_E11.h5')
model_C_E22_E33 = models.load_model('Comp_Prop_E22_E33.h5')
model_C_G12_G13 = models.load_model('Comp_Prop_G12_G13.h5')
model_C_v12_v13 = models.load_model('Comp_Prop_v12_v13.h5')
model_C_v23 = models.load_model('Comp_Prop_v23.h5')

# Load normalization parameters for fiber properties
Xmax_F = np.array([499.9707,74.995605,49.99707,44.990288,3.7497802,1.1249341])
Xmin_F = np.array([80.092735,12.01391,8.009274,0.32069528,0.6006955,0.18020865])
dmax_F_E11, dmin_F_E11 = [499.73843,80.05675]
dmax_F_E22_E33, dmin_F_E22_E33 = [19.78555,3.2063086]
dmax_F_G13_G12, dmin_F_G13_G12 = [13.517561,1.0071378]
dmax_F_v12_v13, dmin_F_v12_v13 = [0.3001687,0.30003452]
dmax_F_v23, dmin_F_v23 = [0.5961686,0.59155756]

# Load normalization parameters for composite properties
Xmax_C = np.array([0.75,500.0,20.0,15.0,0.4,0.7,4.0,0.45])
Xmin_C = np.array([0.3,80.0,2.0,1.0,0.2,0.3,1.0,0.3])
dmax_C_E11, dmin_C_E11 = [370.37097,24.676718]
dmax_C_E22_E33, dmin_C_E22_E33 = [13.923995,1.2882799]
dmax_C_G13_G12, dmin_C_G13_G12 = [6.4756,0.4628]
dmax_C_v12_v13, dmin_C_v12_v13 = [0.44033787,0.21886805]
dmax_C_v23, dmin_C_v23 = [0.78672594,0.3209443]

# ====================================================================================
#              PREDICT PROPERTIES (SHEET -> RADIAL FIBER -> COMPOSITE) 
# ====================================================================================
# Calculate remaining input data
# Note: the nu's for the sheet are held constant at [nu12=nu13=0.3 and nu23=-0.3]
E22_sheet = 0.15*E11_sheet
E33_sheet = 0.1*E11_sheet
G12_sheet = (E22_sheet/E11_sheet)*G12p_sheet
G13_sheet = ((E33_sheet/2.0)/G12p_sheet)*G12_sheet
G23_sheet = 0.3*G13_sheet

# Generate the input vector for radial fiber NN
X_raw = np.array([[E11_sheet,E22_sheet,E33_sheet,G12_sheet,G13_sheet,G23_sheet]])

# Normalize input sheet data
X_F = (X_raw-Xmin_F)/(Xmax_F-Xmin_F)

# Predict FIBER properties
PredictedLabels_F_E11 = model_F_E11.predict(X_F)
ActualPredictedLabels_F_E11 = PredictedLabels_F_E11*(dmax_F_E11-dmin_F_E11)+dmin_F_E11
#print('FIBER E11',ActualPredictedLabels_F_E11)

PredictedLabels_F_E22_E33 = model_F_E22_E33.predict(X_F)
ActualPredictedLabels_F_E22_E33 = PredictedLabels_F_E22_E33*(dmax_F_E22_E33-dmin_F_E22_E33)+dmin_F_E22_E33
#print('FIBER E22=E33',ActualPredictedLabels_F_E22_E33)

PredictedLabels_F_G12_G13 = model_F_G12_G13.predict(X_F)
ActualPredictedLabels_F_G12_G13 = PredictedLabels_F_G12_G13*(dmax_F_G13_G12-dmin_F_G13_G12)+dmin_F_G13_G12
#print('FIBER G12=G13',ActualPredictedLabels_F_G12_G13)

PredictedLabels_F_v12_v13 = model_F_v12_v13.predict(X_F)
ActualPredictedLabels_F_v12_v13 = PredictedLabels_F_v12_v13*(dmax_F_v12_v13-dmin_F_v12_v13)+dmin_F_v12_v13
#print('FIBER v12=v13',ActualPredictedLabels_F_v12_v13)

PredictedLabels_F_v23 = model_F_v23.predict(X_F)
ActualPredictedLabels_F_v23 = PredictedLabels_F_v23*(dmax_F_v23-dmin_F_v23)+dmin_F_v23
#print('FIBER v23',ActualPredictedLabels_F_v23)

# Generate input vector for composite NN
X_raw = np.array([[FVF,ActualPredictedLabels_F_E11[0][0],ActualPredictedLabels_F_E22_E33[0][0],ActualPredictedLabels_F_G12_G13[0][0],ActualPredictedLabels_F_v12_v13[0][0],ActualPredictedLabels_F_v23[0][0],E_matrix,v_matrix]])

# Normalize composite data
X_C = (X_raw-Xmin_C)/(Xmax_C-Xmin_C)

# Predict COMPOSITE properties
PredictedLabels_C_E11 = model_C_E11.predict(X_C)
ActualPredictedLabels_C_E11 = PredictedLabels_C_E11*(dmax_C_E11-dmin_C_E11)+dmin_C_E11
print('COMPOSITE E11 [GPa]:     ',ActualPredictedLabels_C_E11[0][0])

PredictedLabels_C_E22_E33 = model_C_E22_E33.predict(X_C)
ActualPredictedLabels_C_E22_E33 = PredictedLabels_C_E22_E33*(dmax_C_E22_E33-dmin_C_E22_E33)+dmin_C_E22_E33
print('COMPOSITE E22=E33 [GPa]: ',ActualPredictedLabels_C_E22_E33[0][0])

PredictedLabels_C_G12_G13 = model_C_G12_G13.predict(X_C)
ActualPredictedLabels_C_G12_G13 = PredictedLabels_C_G12_G13*(dmax_C_G13_G12-dmin_C_G13_G12)+dmin_C_G13_G12
print('COMPOSITE G12=G13 [GPa]: ',ActualPredictedLabels_C_G12_G13[0][0])

PredictedLabels_C_v12_v13 = model_C_v12_v13.predict(X_C)
ActualPredictedLabels_C_v12_v13 = PredictedLabels_C_v12_v13*(dmax_C_v12_v13-dmin_C_v12_v13)+dmin_C_v12_v13
print('COMPOSITE v12=v13:       ',ActualPredictedLabels_C_v12_v13[0][0])

PredictedLabels_C_v23 = model_C_v23.predict(X_C)
ActualPredictedLabels_C_v23 = PredictedLabels_C_v23*(dmax_C_v23-dmin_C_v23)+dmin_C_v23
print('COMPOSITE v23:           ',ActualPredictedLabels_C_v23[0][0])

# ====================================================================================
#                               DISPLAY WARNINGS 
# ====================================================================================
if E11_sheet < 80.0 or E11_sheet > 500.0:
    print('WARNING: Input (Sheet E11) is out-of-range. Final result has increased error.')
if G12p_sheet < 1.0 or G12p_sheet > 300.0:
    print('WARNING: Input (Sheet G12) is out-of-range. Final result has increased error.')
if FVF < 0.3 or FVF > 0.75:
    print('WARNING: Input (FVF) is out-of-range. Final result has increased error.')
if E_matrix < 1.0 or E_matrix > 4.0:
    print('WARNING: Input (Matrix E) is out-of-range. Final result has increased error.')
if v_matrix < 0.3 or v_matrix > 0.45:
    print('WARNING: Input (Matrix v) is out-of-range. Final result has increased error.')



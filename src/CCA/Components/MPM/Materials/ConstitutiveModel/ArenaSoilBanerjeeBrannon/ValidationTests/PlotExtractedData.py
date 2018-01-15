#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import csv
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
import scipy
from scipy import linalg


# List of tests
ARENA_PART_SAT_TEST_LIST = [
   'MasonSandUniaxialStrainSHPB_26.ups',
   'MasonSandUniaxialStrainSHPB-081612-003.ups',
   'MasonSandUniaxialStrainSHPBwDamage_26.ups',
   'MasonSandUniaxialStrainSHPBwDisagg_26.ups',
   'BoulderClayDryUniaxialStrainSHPB_Mean.ups',
   'BoulderClayUniaxialStrainSHPB-072313-001.ups',
   'MasonSandSHPB052212-001.ups',
   'MasonSandSHPB052212-006.ups',
   'MasonSandSHPB052212-011.ups',
   'MasonSandSHPB052212-016.ups',
   'BoulderClaySHPB072213-014.ups',
   'BoulderClaySHPB072313-013.ups',
   'BoulderClayDrySHPB-Erik.ups',
   'BoulderClayDrySHPB-oldK-Erik.ups',
   'BoulderClaySHPB072313-001.ups',
]

# List of experimental data
SHPB_EXPT_DATA_LIST = [
  'Masonsand052212-026.expt',
  'Masonsand081612-003.expt',
  'Masonsand052212-026.expt',
  'Masonsand052212-026.expt',
  'BoulderClaySHPBDry.expt',
  'BoulderClay072313-001.expt',
  'Masonsand052212-001.expt',
  'Masonsand052212-006.expt',
  'Masonsand052212-011.expt',
  'Masonsand052212-016.expt',
  'BoulderClay072213-014.expt',
  'BoulderClay072313-013.expt',
  'BoulderClaySHPBDry.expt',
  'BoulderClaySHPBDry.expt',
  'BoulderClay072313-001.expt',
]

# Zip the two lists
ZIPPED_LIST = zip(ARENA_PART_SAT_TEST_LIST, SHPB_EXPT_DATA_LIST)

### COMMENT ME OUT!!!!!!! ###
ARENA_PART_SAT_TEST_LIST = [
  #ZIPPED_LIST[0],  # SHPB: sample 26
  #ZIPPED_LIST[1],  # SHPB 18 Wet: sample 3
  ZIPPED_LIST[2],  # SHPB: sample 26 with damage 
  #ZIPPED_LIST[3],  # SHPB: sample 26 with disaggregation 
  #ZIPPED_LIST[4],  # SHPB: Boulder clay: dry : mean
  #ZIPPED_LIST[5],  # SHPB: Boulder clay: Fully saturated
  #ZIPPED_LIST[6],  # SHPB: Mason sand: 1770 kg/m^3
  #ZIPPED_LIST[7],  # SHPB: Mason sand: 1700 kg/m^3
  #ZIPPED_LIST[8],  # SHPB: Mason sand: 1580 kg/m^3
  #ZIPPED_LIST[9],  # SHPB: Mason sand: 1640 kg/m^3
  #ZIPPED_LIST[10],  # SHPB: Boulder clay: 12.8% w/w
  #ZIPPED_LIST[11],  # SHPB: Boulder clay: 40.8% w/w
  #ZIPPED_LIST[12],  # SHPB: Boulder clay: dry : mean: Erik's parameters
  #ZIPPED_LIST[13],  # SHPB: Boulder clay: dry : mean: Erik's parameters
  #ZIPPED_LIST[14],  # SHPB: Boulder clay:  40.8% w/w
  ]
### --------------------- ###

# Assume that uda files are in this directory and links to executables are
# in the parent directory
root_dir = os.path.abspath(".")

# Print the tests to be extracted
for test in ARENA_PART_SAT_TEST_LIST:
  print test

# Set default plot parameters
def setPlotParams():
  fontSize = 12
  lineWidth = 2
  markers = None
  plt.rcParams['legend.loc'] = 'best'
  plt.rcParams['mathtext.it'] = 'serif:italic'
  plt.rcParams['mathtext.rm'] = 'serif:bold'
  plt.rcParams['mathtext.sf'] = 'sans'
  plt.rcParams['font.size'] = fontSize
  plt.rcParams['font.weight'] = 'bold'
  plt.rcParams['axes.labelsize'] = 'medium'
  plt.rcParams['axes.labelweight'] = 'bold'
  plt.rcParams['legend.fontsize'] = 'medium'
  plt.rcParams['lines.linewidth'] = lineWidth
  plt.rcParams['lines.markersize'] = 8
  plt.rcParams['xtick.major.pad']  = 12
  plt.rcParams['ytick.major.pad']  = 8
  plt.rcParams['xtick.major.size'] = 6
  plt.rcParams['xtick.minor.size'] = 3
  plt.rcParams['ytick.major.size'] = 6
  plt.rcParams['ytick.minor.size'] = 3
  plt.rcParams['figure.dpi'] = 120
  font = {'family' : 'normal',
          'weight' : 'bold',
          'size'   : fontSize}
  plt.rc('font', **font)
  #plt.rc('text', usetex=True)
  #plt.rc('text.latex', preamble=r'\usepackage{sansmath}')
  
  return

def savePNG(name,size='1920x1080'):
  res = float(plt.rcParams['figure.dpi'])
  #Add Check for file already existing as name.png
  if size == '640x480':
    size = [640/res,480/res]
  if size == '1080x768':
    size = [1080/res,768/res]
  if size == '1152x768':
    size = [1152/res,768/res]
  if size == '1280x854':
    size = [1280/res,854/res]
  if size == '1280x960':
    size = [1280/res,960/res]
  if size == '1920x1080':
    size = [1920/res,1080/res]
  #set the figure size for saving
  plt.gcf().set_size_inches(size[0],size[1])
  #save at speciified resolution
  plt.savefig(name+'.png', bbox_inches=0, dpi=plt.rcParams['figure.dpi'])

def savePDF(name,size='1920x1080'):
  res = float(plt.rcParams['figure.dpi'])
  #Add Check for file already existing as name.png
  if size == '640x480':
    size = [640/res,480/res]
  if size == '1080x768':
    size = [1080/res,768/res]
  if size == '1152x768':
    size = [1152/res,768/res]
  if size == '1280x854':
    size = [1280/res,854/res]
  if size == '1280x960':
    size = [1280/res,960/res]
  if size == '1920x1080':
    size = [1920/res,1080/res]
  #set the figure size for saving
  plt.gcf().set_size_inches(size[0],size[1])
  #save at speciified resolution
  plt.savefig(name+'.pdf', bbox_inches=0, dpi=plt.rcParams['figure.dpi'])

#---------------------------------------------------------------------------
# Read the extracted particle variable data
#---------------------------------------------------------------------------
def getVarData(data_row, VARIABLE_TYPE):
  result = {
    'time'  : lambda data_row: [data_row[i] for i in range(0,1)],
    'scalar': lambda data_row: [data_row[i] for i in range(4,5)] , 
    'vector': lambda data_row: [data_row[i] for i in range(4,7)] ,
    'matrix': lambda data_row: [data_row[i] for i in range(4,13)]
  }[VARIABLE_TYPE](data_row)

  return result

def readCSVFile(file_name, VARIABLE_TYPE):
  x_data = []
  with open(file_name, 'rb') as x_file:
    reader = csv.reader(x_file, delimiter=" ")
    for row in reader:
      if not row:
        pass
      else: 
        row_data = getVarData(row, VARIABLE_TYPE)
        x_data.append(row_data)
  return x_data

#---------------------------------------------------------------------------
# Convert the data into a scalar
#---------------------------------------------------------------------------
def matrix_det(mat):
  return np.linalg.det(mat)

def matrix_iso(mat):
  return (np.trace(mat)/3.0)*np.eye(3)

def matrix_dev(mat):
  return mat - matrix_iso(mat)

def matrix_I1(mat):
  return mat.trace()

def matrix_J2(mat):
  return 0.5*np.dot(matrix_dev(mat), matrix_dev(mat)).trace()

def matrix_J3(mat):
  return (1/3.0)*np.dot(np.dot(matrix_dev(mat), matrix_dev(mat)), matrix_dev(mat)).trace()

def computeDeterminant(data, VARIABLE_TYPE):
  result = {
    'scalar': lambda data: data[0],
    'vector': lambda data: 0.0,
    'matrix': lambda data: matrix_det(np.reshape(data, (3,3)))
  }[VARIABLE_TYPE](data)
  return result

def computeI1(data, VARIABLE_TYPE):
  result = {
    'scalar': lambda data: data[0],
    'vector': lambda data: np.mean(data), 
    'matrix': lambda data: matrix_I1(np.reshape(data, (3,3)))
  }[VARIABLE_TYPE](data)
  return result

def computeJ2(data, VARIABLE_TYPE):
  result = {
    'scalar': lambda data: data[0], 
    'vector': lambda data: 0.0, 
    'matrix': lambda data: np.sign(matrix_J3(np.reshape(data, (3,3))))*
                                matrix_J2(np.reshape(data, (3,3)))
  }[VARIABLE_TYPE](data)
  return result

def getComponent(data, index, VARIABLE_TYPE):
  result = {
    'scalar': lambda data: data[0], 
    'vector': lambda data: data[max(index,3)], 
    'matrix': lambda data: data[index]
  }[VARIABLE_TYPE](data)
  return result

def computeStretch(data):
  R, U = scipy.linalg.polar(np.reshape(data, (3,3)))
  eigs = np.linalg.eig(U)[0]
  return eigs
  
def getStretch(data, index, VARIABLE_TYPE):
  result = {
    'scalar': lambda data: 0.0,
    'vector': lambda data: 0.0,
    'matrix': lambda data: computeStretch(data)[index]
  }[VARIABLE_TYPE](data)
  return result

def convertData(data, VARIABLE_TYPE, CONVERSION_TYPE):
  data = data.astype(np.float)
  result = {
    'none': lambda data: data[0], 
    'det': lambda data: computeDeterminant(data, VARIABLE_TYPE),
    'I1': lambda data: computeI1(data, VARIABLE_TYPE),
    'J2': lambda data: computeJ2(data, VARIABLE_TYPE),
    '11': lambda data: getComponent(data, 0, VARIABLE_TYPE),
    '12': lambda data: getComponent(data, 1, VARIABLE_TYPE),
    '22': lambda data: getComponent(data, 4, VARIABLE_TYPE),
    'lambda1': lambda data: getStretch(data, 0, VARIABLE_TYPE),
    'lambda2': lambda data: getStretch(data, 1, VARIABLE_TYPE),
    'lambda3': lambda data: getStretch(data, 2, VARIABLE_TYPE)
  }[CONVERSION_TYPE](data)

  return result

#---------------------------------------------------------------------------
# Read extracted data and converted into scalar particle variables 
#---------------------------------------------------------------------------
def readExtractedPartVar(input_dir, test_name, particle_variable, 
                         VARIABLE_TYPE = "scalar", CONVERSION_TYPE = "none"):

  print "Variable: ", particle_variable

  # Extract the particle variable name
  var_name = particle_variable.split('p.')[1]

  # Create a particle variable input filename
  input_file_name = os.path.join(input_dir, var_name + ".out_p01")

  # Read the files
  data = readCSVFile(input_file_name, VARIABLE_TYPE)

  # Convert data if needed
  data = map(lambda x: convertData(np.asarray(x), VARIABLE_TYPE, CONVERSION_TYPE), 
             data)

  return data

#---------------------------------------------------------------------------
# Read the extracted the data from a test
#---------------------------------------------------------------------------
def read_extracted_test_data(root_dir, ups_file):
  ''' '''
  print '\nPlotting extracted test data from :\t', ups_file

  # Find the test name
  test_name = ups_file.split(".ups")[0]
  print test_name

  # Find the input directory name
  input_dir = os.path.join(root_dir, test_name)
  print input_dir

  # Change current working directory to the output directory
  os.chdir(input_dir)

  # Read the time
  time = readExtractedPartVar(input_dir, test_name, "p.deformationMeasure", "time", "none")

  # Read deformation gradient
  detF = readExtractedPartVar(input_dir, test_name, "p.deformationMeasure", "matrix", "det")
  F_11 = readExtractedPartVar(input_dir, test_name, "p.deformationMeasure", "matrix", "11")
  F_22 = readExtractedPartVar(input_dir, test_name, "p.deformationMeasure", "matrix", "22")
  lambda_1 = readExtractedPartVar(input_dir, test_name, "p.deformationMeasure", "matrix", "lambda1")
  eps_11 = [1.0 + F for F in F_11]
  eps_22 = [1.0 + F for F in F_22]

  # Read stress
  sig_I1 = readExtractedPartVar(input_dir, test_name, "p.stress", "matrix", "I1")
  sig_J2 = readExtractedPartVar(input_dir, test_name, "p.stress", "matrix", "J2")
  sig_11 = readExtractedPartVar(input_dir, test_name, "p.stress", "matrix", "11")
  sig_22 = readExtractedPartVar(input_dir, test_name, "p.stress", "matrix", "22")
  sig_12 = readExtractedPartVar(input_dir, test_name, "p.stress", "matrix", "12")

  # Read quasistatic stress
  sigQS_I1 = readExtractedPartVar(input_dir, test_name, "p.stressQS", "matrix", "I1")
  sigQS_J2 = readExtractedPartVar(input_dir, test_name, "p.stressQS", "matrix", "J2")

  # Read capX
  capX = readExtractedPartVar(input_dir, test_name, "p.capX", "scalar", "none")

  # Read plastic strain
  epsp_I1 = readExtractedPartVar(input_dir, test_name, "p.plasticStrain", "matrix", "I1")

  # Read elastic volumetric strain
  epse_v = readExtractedPartVar(input_dir, test_name, "p.elasticVolStrain", "scalar", "none")

  # Read plastic volumetric strain
  epsp_v = readExtractedPartVar(input_dir, test_name, "p.plasticVolStrain", "scalar", "none")

  # Read equivalent plastic strain
  epsp_eq = readExtractedPartVar(input_dir, test_name, "p.plasticCumEqStrain", "scalar", "none")

  # Read pore pressure
  alpha_I1 = readExtractedPartVar(input_dir, test_name,  "p.porePressure", "matrix", "I1")

  # Read porosity
  phi = readExtractedPartVar(input_dir, test_name, "p.porosity", "scalar", "none")

  # Read saturation
  sw = readExtractedPartVar(input_dir, test_name, "p.saturation", "scalar", "none")

  # Read coherence
  coher = readExtractedPartVar(input_dir, test_name, "p.COHER", "scalar", "none")

  # Read t_grow
  tgrow = readExtractedPartVar(input_dir, test_name, "p.TGROW", "scalar", "none")

  # Read yield surface parameters
  PEAKI1 = readExtractedPartVar(input_dir, test_name, "p.ArenaPEAKI1", "scalar", "none")
  FSLOPE = readExtractedPartVar(input_dir, test_name, "p.ArenaFSLOPE", "scalar", "none")
  STREN = readExtractedPartVar(input_dir, test_name, "p.ArenaSTREN", "scalar", "none")
  YSLOPE = readExtractedPartVar(input_dir, test_name, "p.ArenaYSLOPE", "scalar", "none")

  print('Reading and conversion done.')  

  # Save the read data as a named tuple
  ReadData = collections.namedtuple('ReadData', ['time', 'detF', 'F_11', 'F_22', 'lambda_1', 
    'sig_I1', 'sig_J2', 'sig_11', 'sig_12', 'sig_22', 'sigQS_I1', 'sigQS_J2', 
    'capX', 'epsp_I1', 'epse_v', 'epsp_v', 'epsp_eq', 'alpha_I1', 'phi', 'sw',
    'coher', 'tgrow', 'PEAKI1', 'FSLOPE', 'STREN', 'YSLOPE'])

  return ReadData(time, detF, F_11, F_22, lambda_1, 
                  sig_I1, sig_J2, sig_11, sig_12, sig_22,
                  sigQS_I1, sigQS_J2, capX, epsp_I1, epse_v, epsp_v, epsp_eq,
                  alpha_I1, phi, sw, coher, tgrow, 
                  PEAKI1, FSLOPE, STREN, YSLOPE)

#---------------------------------------------------------------------------
# Read the experimental SHPB data
#---------------------------------------------------------------------------
def read_expt_data(root_dir, expt_file):

  # Create expt file path
  file_name_expt = os.path.join(root_dir, expt_file)
  print("Reading experimental data from", file_name_expt)

  # Open the experimental data file
  I1_expt = []
  J2_expt = []
  J3_expt = []
  time_expt = []
  eps_a_expt = []
  sigma_a_expt = []
  sigma_r_expt = []
  sigma_m_expt = []
  with open(file_name_expt, 'rb') as exptfile:
    reader = csv.DictReader(exptfile, delimiter=' ', quotechar='"')
    for row in reader:
      #print(row['Axial stress (computed)'], row['mean stress (computed)']) 
      time = float(row['Time'])
      eps_a = float(row['Axial strain(computed)'])
      sigma_a = float(row['Axial stress (computed)'])
      sigma_m = float(row['mean stress (computed)'])
      #print("sig_a = ", sigma_a, " sig_m = ", sigma_m) 
      #print(type(sigma_a).__name__, type(sigma_m).__name__) 

      sigma_r = 0.5*(3.0*sigma_m - sigma_a)
      I1 = 3.0*sigma_m
      J2 = pow(sigma_a - sigma_r, 2)/3.0
      J3 = pow(sigma_a - sigma_r, 3)*2.0/27.0
      #print("I1 = ", I1, " J2 = ", J2, " J3 = ", J3) 

      time_expt.append(time)
      eps_a_expt.append(eps_a)
      sigma_a_expt.append(sigma_a)
      sigma_r_expt.append(sigma_r)
      sigma_m_expt.append(sigma_m)
      I1_expt.append(I1)
      J2_expt.append(J2)
      J3_expt.append(J3)

  p_expt = -np.array(I1_expt)/3.0
  q_expt = -np.sign(J3_expt)*np.sqrt(3.0*np.array(J2_expt))

  z_expt = -np.array(I1_expt)/np.sqrt(3.0)
  r_expt = -np.sign(J3_expt)*np.sqrt(2.0*np.array(J2_expt))

  # Save the read data as a named tuple
  ExptData = collections.namedtuple('ExptData', ['time', 'eps_11', 'sig_11', 'sig_22', 'sig_m',
    'sig_I1', 'sig_J2', 'sig_J3', 'p', 'q', 'z', 'r'])
  return ExptData(time_expt, eps_a_expt, sigma_a_expt, sigma_r_expt, sigma_m_expt,
                  I1_expt, J2_expt, J3_expt, p_expt, q_expt, z_expt, r_expt)
  

#---------------------------------------------------------------------------
# Plot two scalar particle variables from the extracted particle variable data
#---------------------------------------------------------------------------
def convertAndScaleData(data, OPERATION_TYPE, scale_fator):
  data = data.astype(np.float)
  result = {
    'none': lambda data: data, 
    'log': lambda data: np.log(data),
    'p': lambda data: data/3.0,
    'q': lambda data: np.sign(data)*np.sqrt(3.0*abs(data)),
    'z': lambda data: data/np.sqrt(3.0),
    'r': lambda data: np.sign(data)*np.sqrt(2.0*abs(data))
  }[OPERATION_TYPE](data)

  result = result*scale_fator

  return result

def plotExtractedPartVars(test_name, pdf_name,
                          x_data, X_OPERATION, x_scale, X_VARIABLE_LABEL, 
                          y_data, Y_OPERATION, y_scale, Y_VARIABLE_LABEL): 

  print "Variables: ", X_VARIABLE_LABEL, ", ", Y_VARIABLE_LABEL

  # Convert and scale data if needed
  x_data = map(lambda xx: convertAndScaleData(np.asarray(xx), X_OPERATION, x_scale), 
               x_data)
  y_data = map(lambda yy: convertAndScaleData(np.asarray(yy), Y_OPERATION, y_scale), 
               y_data)

  # Plot the data
  #setPlotParams()
  #plt.rcdefaults()
  fig = plt.figure(figsize=(6,6))
  #ax1 = plt.subplot(111)
  x_data = np.array(x_data)
  y_data = np.array(y_data)
  plt.quiver(x_data[:-1], y_data[:-1], 
             x_data[1:]-x_data[:-1], y_data[1:]-y_data[:-1], 
             scale_units='xy', angles='xy', scale=1.0, width=0.001, 
             headwidth=7, headlength=10,
             linewidth=0.5, edgecolor='b', color='k')
  #plt.plot(x_data, y_data, 'b-')
  plt.xlabel(X_VARIABLE_LABEL, fontsize = 16)
  plt.ylabel(Y_VARIABLE_LABEL, fontsize = 16)
  plt.title(test_name, fontsize = 16)
  #savePDF(pdf_name)
  plt.savefig(pdf_name + ".pdf", bbox_inches = 'tight')
  #plt.show()
  plt.close(fig)

  return

#---------------------------------------------------------------------------
# Plot the extracted and converted data from a test
#---------------------------------------------------------------------------
def plot_extracted_test_data(ups_file, data, stress_scale_fac):
  ''' '''
  print '\nPlotting extracted test data :\t'

  # Move to a smaller var name and get units
  fac = stress_scale_fac
  units = "(MPa)"
  if (fac < 1.0e-6):
    units = "(GPa)"

  # Find the test name
  test_name = ups_file.split(".ups")[0]
  
  # Set plot defaults
  fontSize = 14
  lineWidth = 2
  plt.rcParams['font.size'] = fontSize
  plt.rcParams['font.weight'] = 'normal'
  plt.rcParams['lines.linewidth'] = lineWidth
  plt.rcParams['mathtext.fontset'] = 'stixsans'

  # Plot time vs deformation gradient 
  plotExtractedPartVars(test_name, test_name + "_F11_t",
                        data.time, "none", 1.0e6, "$\\mathbf{t}$ " + "$\\mu$s", 
                        data.F_11, "none", -1.0, "$\\mathbf{-F_{11}}$") 

  # Plot time vs stress
  plotExtractedPartVars(test_name, test_name + "_S11_t",
                        data.time, "none", 1.0e6, "$\\mathbf{t}$ " + "$\\mu$s", 
                        data.sig_11, "none", -fac, "$\\mathbf{\\sigma_{11}}$ "+units)
  plotExtractedPartVars(test_name, test_name + "_S22_t",
                        data.time, "none", 1.0e6, "$\\mathbf{t}$ " + "$\\mu$s", 
                        data.sig_22, "none", -fac, "$\\mathbf{\\sigma_{22}}$ "+units)

  # Plot def grad vs stress
  plotExtractedPartVars(test_name, test_name + "_S22_F11",
                        data.F_11, "none", -1.0, "$\\mathbf{-F_{11}}$", 
                        data.sig_22, "none", -fac, "$\\mathbf{\\sigma_{22}}$ "+units)
  plotExtractedPartVars(test_name, test_name + "_S12_F11",
                        data.F_11, "none", -1.0, "$\\mathbf{-F_{11}}$", 
                        data.sig_12, "none", -fac, "$\\mathbf{\\sigma_{12}}$ "+units)
  plotExtractedPartVars(test_name, test_name + "_S11_lambda1",
                        data.lambda_1, "none", -1.0, "$\\mathbf{\\lambda_1}$", 
                        data.sig_11, "none", -fac, "$\\mathbf{\\sigma_{11}}$ "+units)

  # Plot stress vs deformation gradient 
  plotExtractedPartVars(test_name, test_name + "_S11_F11",
                        data.F_11, "none", -1.0, "$\\mathbf{-F_{11}}$", 
                        data.sig_11, "none", -1/3*fac, "$\\mathbf{\\sigma_{11}}$ "+units)
  plotExtractedPartVars(test_name, test_name + "_S22_F11",
                        data.F_11, "none", -1.0, "$\\mathbf{-F_{11}}$", 
                        data.sig_22, "none", -1/3*fac, "$\\mathbf{\\sigma_{22}}$ "+units)
  plotExtractedPartVars(test_name, test_name + "_S12_F11",
                        data.F_11, "none", -1.0, "$\\mathbf{-F_{11}}$", 
                        data.sig_12, "none", -1/3*fac, "$\\mathbf{\\sigma_{12}}$ "+units)
  plotExtractedPartVars(test_name, test_name + "_S11_lambda1",
                        data.lambda_1, "none", -1.0, "$\\mathbf{\\lambda_1}$", 
                        data.sig_11, "none", -1/3*fac, "$\\mathbf{\\sigma_{11}}$ "+units)

  # Plot stress vs strain
  plotExtractedPartVars(test_name, test_name + "_sig_m_eps_v",
                        data.detF, "log", -1.0, "$\\mathbf{\\varepsilon_v}$", 
                        data.sig_I1, "none", -1/3*fac, "$\\mathbf{\\sigma_m}$ "+units)

  # Plot capX 
  plotExtractedPartVars(test_name, test_name + "_capX_eps_p_v",
                        data.epsp_v, "none", -1.0, "$\\mathbf{\\varepsilon_v^p}$", 
                        data.capX, "none", -1.0*fac, "$\\mathbf{\\bar{X}}$ "+units)
                
  # Plot plastic strain 
  plotExtractedPartVars(test_name, test_name + "_sig_m_eps_p_v_a",
                        data.epsp_I1, "none", -1.0, "$\\mathbf{\\varepsilon_v^p}$", 
                        data.sig_I1, "none", -1/3*fac, "$\\mathbf{\\sigma_m}$ "+units)

  # Plot elastic volumetric strain 
  plotExtractedPartVars(test_name, test_name + "_sig_m_eps_e_v",
                        data.epse_v, "none", -1.0, "$\\mathbf{\\varepsilon_v^e}$", 
                        data.sig_I1, "none", -1/3*fac, "$\\mathbf{\\sigma_m}$ "+units)

  # Plot plastic volumetric strain 
  plotExtractedPartVars(test_name, test_name + "_sig_m_eps_p_v",
                        data.epsp_v, "none", -1.0, "$\\mathbf{\\varepsilon_v^p}$", 
                        data.sig_I1, "none", -1/3*fac, "$\\mathbf{\\sigma_m}$ "+units)

  # Plot plastic equivalent strain 
  plotExtractedPartVars(test_name, test_name + "_sig_m_eps_p_eq",
                        data.epsp_eq, "none", 1.0, "$\\mathbf{\\varepsilon_{eq}^p}$", 
                        data.sig_I1,  "none", -1/3*fac, "$\\mathbf{\\sigma_m}$ "+units)

  # Plot pore pressure 
  plotExtractedPartVars(test_name, test_name + "_pbarw_eps_p_v",
                        data.epsp_v,   "none", -1.0, "$\\mathbf{\\varepsilon_v^p}$", 
                        data.alpha_I1, "none", -1.0/3*1.0e-6, "$\\mathbf{\\bar{p^w}}$ (MPa)")

  # Plot porosity
  plotExtractedPartVars(test_name, test_name + "_phi_eps_p_v",
                        data.epsp_v, "none", -1.0, "$\\mathbf{\\varepsilon_v^p}$", 
                        data.phi,    "none",  1.0, "$\\mathbf{\\phi}$")

  # Plot saturation
  plotExtractedPartVars(test_name, test_name + "_sw_eps_p_v",
                        data.epsp_v, "none", -1.0, "$\\mathbf{\\varepsilon_v^p}$", 
                        data.sw,     "none",  1.0, "$\\mathbf{S_w}$")

  # Plot coherence
  plotExtractedPartVars(test_name, test_name + "_coher_t",
                        data.time,  "none", 1.0e6, "$\\mathbf{t}$ " + "$\\mu$s", 
                        data.coher, "none", 1.0, "$\\mathbf{Coherence}$")

  # Plot t_grow
  plotExtractedPartVars(test_name, test_name + "_tgrow_t",
                        data.time,  "none", 1.0e6, "$\\mathbf{t}$ " + "$\\mu$s", 
                        data.tgrow, "none", 1.0, "$\\mathbf{t_{grow}}$")

  print('Plotting extracted data done.')  

  return 

#---------------------------------------------------------------------------
# Plot the extracted stress and yield surface
#---------------------------------------------------------------------------
def str_to_mathbf(string):
  #Only works with single spaces no leading space
  string = string.split()
  return_string = ''
  for elem in string:
    elem = r'$\mathbf{'+elem+'}$'
    return_string+=elem+'  '
  return return_string[0:-1]

def getMaterialParameters(ups_file):

  # Open ups file for reading
  ups_file_path = os.path.join(root_dir, ups_file)
  ups_file_id = open(ups_file_path, "r")

  # Read
  check_lines = False
  already_read = False
  material_dict = {}
  for line in ups_file_id:
    if '<constitutive_model' in line and 'type' in line and '"ArenaSoil"' in line and not(already_read):
      check_lines = True
    if check_lines and not(already_read):
      if '<initial_porosity>' in line:
        material_dict['phi0']  = float(line.split('<initial_porosity>')[1].split('</initial_porosity>')[0].strip())                      
      if '<initial_saturation>' in line:
        material_dict['sw0']  = float(line.split('<initial_saturation>')[1].split('</initial_saturation>')[0].strip())                      
      if '<initial_fluid_pressure>' in line:
        material_dict['pf0']  = float(line.split('<initial_fluid_pressure>')[1].split('</initial_fluid_pressure>')[0].strip())                      
      if '<PEAKI1>' in line:
        material_dict['PEAKI1'] = float(line.split('<PEAKI1>')[1].split('</PEAKI1>')[0].strip())
      if '<FSLOPE>' in line:
        material_dict['FSLOPE'] = float(line.split('<FSLOPE>')[1].split('</FSLOPE>')[0].strip())
      if '<STREN>' in line:
        material_dict['STREN'] = float(line.split('<STREN>')[1].split('</STREN>')[0].strip())
      if '<YSLOPE>' in line:
        material_dict['YSLOPE'] = float(line.split('<YSLOPE>')[1].split('</YSLOPE>')[0].strip()) 
      if '<BETA>' in line:
        material_dict['BETA'] = float(line.split('<BETA>')[1].split('</BETA>')[0].strip()) 
      if '<b0>' in line:
        material_dict['b0'] = float(line.split('<b0>')[1].split('</b0>')[0].strip())
      if '<b1>' in line:
        material_dict['b1'] = float(line.split('<b1>')[1].split('</b1>')[0].strip())
      if '<b2>' in line:
        material_dict['b2'] = float(line.split('<b2>')[1].split('</b2>')[0].strip())
      if '<b3>' in line:
        material_dict['b3'] = float(line.split('<b3>')[1].split('</b3>')[0].strip())
      if '<b4>' in line:
        material_dict['b4'] = float(line.split('<b4>')[1].split('</b4>')[0].strip())
      if '<G0>' in line:
        material_dict['g0'] = float(line.split('<G0>')[1].split('</G0>')[0].strip())
      if '<nu1>' in line:
        material_dict['nu1'] = float(line.split('<nu1>')[1].split('</nu1>')[0].strip())
      if '<nu2>' in line:
        material_dict['nu2'] = float(line.split('<nu2>')[1].split('</nu2>')[0].strip())
      if '<p0>' in line:
        material_dict['p0'] = float(line.split('<p0>')[1].split('</p0>')[0].strip())
      if '<p1>' in line:
        material_dict['p1'] = float(line.split('<p1>')[1].split('</p1>')[0].strip())        
      if '<p1_sat>' in line:
        material_dict['p1_sat'] = float(line.split('<p1_sat>')[1].split('</p1_sat>')[0].strip())
      if '<p2>' in line:
        material_dict['p2'] = float(line.split('<p2>')[1].split('</p2>')[0].strip()) 
      if '<p3>' in line:
        material_dict['p3'] = float(line.split('<p3>')[1].split('</p3>')[0].strip()) 
      if '<CR>' in line:
        material_dict['CR'] = float(line.split('<CR>')[1].split('</CR>')[0].strip())
      if '<T1>' in line:
        material_dict['T1']  = float(line.split('<T1>')[1].split('</T1>')[0].strip())
      if '<T2>' in line:
        material_dict['T2']  = float(line.split('<T2>')[1].split('</T2>')[0].strip())
      if '<subcycling_characteristic_number>' in line:
        material_dict['nsub']  = float(line.split('<subcycling_characteristic_number>')[1].split('</subcycling_characteristic_number>')[0].strip())                
      if '</constitutive_model>' in line:
        already_read = True
        check_lines = False
  ups_file_id.close()
      
  # Organize the dictionary in printable form
  tmp_string = r'Material Properties:'+'\n'
  key_list = material_dict.keys()
  key_list.sort()
  for key in key_list:
    if '_' in key:
      tmp = key.split('_')
      tmp = str_to_mathbf(tmp[0]+'_'+'{'+tmp[1]+'}')
      tmp_string += tmp+str_to_mathbf(' = ')+str_to_mathbf(format(material_dict[key],'1.3e'))+'\n'
    else:
      tmp = key
      if key == 'nsub':
        tmp_string += str_to_mathbf(tmp+' = '+format(material_dict[key],'4.1f'))+'\n'
      else:
        tmp_string += str_to_mathbf(tmp+' = '+format(material_dict[key],'1.3e'))+'\n'
  material_dict['legend'] = tmp_string[0:-1]

  print '--Material parameters--'
  for key in material_dict:
    print key,':',material_dict[key]

  return material_dict

def computeAndPlotYieldSurface(yieldParams, time,
                               PEAKI1, FSLOPE, STREN, YSLOPE,
                               capX, alpha_I1, 
                               X_OPERATION, x_scale, Y_OPERATION, y_scale, COLOR,
                               pdf_name):

  # Get yield parameters
  #PEAKI1 = yieldParams['PEAKI1']
  #FSLOPE = yieldParams['FSLOPE']
  #STREN =  yieldParams['STREN']
  #YSLOPE = yieldParams['YSLOPE']
  CR =     yieldParams['CR']
  pf0 =    yieldParams['pf0']

  # Set up constants
  a1 = STREN
  a2 = (FSLOPE-YSLOPE)/(STREN-YSLOPE*PEAKI1)
  a3 = (STREN-YSLOPE*PEAKI1)*np.exp(-a2*PEAKI1)
  a4 = YSLOPE

  # Compute kappa
  X_eff = capX - alpha_I1
  kappa = PEAKI1 - CR*(PEAKI1 - X_eff)

  # Create an array of I1_eff values
  num_points = 200
  I1_eff_list = np.linspace(0.9999*X_eff, PEAKI1, num_points)
  J2_list = []

  # Compute J2 versus I1
  for I1_eff in I1_eff_list:

    # Compute F_f
    Ff = a1 - a3*np.exp(a2*I1_eff) - a4*(I1_eff)
    Ff_sq = Ff**2

    # Compute Fc
    Fc_sq = 1.0
    if (I1_eff < kappa) and (X_eff< I1_eff):
      ratio = (kappa - I1_eff)/(kappa - X_eff)
      Fc_sq = 1.0 - ratio**2

    # Compute J2
    J2 = Ff_sq*Fc_sq
    J2_list.append(J2)

  # Convert back to I1
  I1_list = I1_eff_list + alpha_I1

  # Convert and scale data if needed
  x_data = map(lambda xx: convertAndScaleData(np.asarray(xx), X_OPERATION, x_scale), 
               I1_list)
  y_data = map(lambda yy: convertAndScaleData(np.asarray(yy), Y_OPERATION, y_scale), 
               J2_list)
  y_data_neg = map(lambda yy: convertAndScaleData(np.asarray(yy), Y_OPERATION, -y_scale), 
               J2_list)

  # Plot the yield surface
  line1 = plt.plot(x_data,  y_data, '-b', linewidth = 0.5)
  line2 = plt.plot(x_data,  y_data_neg, '-b', linewidth = 0.5)
  plt.setp(line1, color=COLOR)
  plt.setp(line2, color=COLOR)

  # Write the data
  outputfile = pdf_name + ".dat"
  yieldLabel = "YieldSurface"
  timeLabel = str(time)
  with open(outputfile, "a") as outfile:
    writer = csv.writer(outfile)
    for x, y in zip(x_data, y_data):
      writer.writerow([timeLabel, x, y, yieldLabel])
    for x, y in zip(x_data, y_data_neg):
      writer.writerow([timeLabel, x, y, yieldLabel])

def plotExtractedStress(test_name, pdf_name, sim_data,
                        I1_data, X_OPERATION, x_scale, X_VARIABLE_LABEL, 
                        J2_data, Y_OPERATION, y_scale, Y_VARIABLE_LABEL,
                        capX_data, alpha_I1_data, yieldParams): 

  print "Variables: ", X_VARIABLE_LABEL, ", ", Y_VARIABLE_LABEL

  # Convert and scale data if needed
  x_data = map(lambda xx: convertAndScaleData(np.asarray(xx), X_OPERATION, x_scale), 
               I1_data)
  y_data = map(lambda yy: convertAndScaleData(np.asarray(yy), Y_OPERATION, y_scale), 
               J2_data)

  # Choose a paired colormap
  colors = map(lambda ii: cm.Paired(ii),
               np.linspace(0.0, 1.0, len(x_data)))

  # Create a figure
  fig = plt.figure(figsize=(8,6))
  res = map(lambda X, alpha, COLOR, time, PEAKI1, FSLOPE, STREN, YSLOPE : 
            computeAndPlotYieldSurface(yieldParams, time,
                                       PEAKI1, FSLOPE, STREN, YSLOPE,
                                       X, alpha, 
                                       X_OPERATION, x_scale,  
                                       Y_OPERATION, y_scale, COLOR, pdf_name),
            capX_data, alpha_I1_data, colors, sim_data.time,
            sim_data.PEAKI1, sim_data.FSLOPE, sim_data.STREN, sim_data.YSLOPE)

  #ax1 = plt.subplot(111)
  x_data = np.array(x_data)
  y_data = np.array(y_data)
  plt.quiver(x_data[:-1], y_data[:-1], 
             x_data[1:]-x_data[:-1], y_data[1:]-y_data[:-1], 
             scale_units='xy', angles='xy', scale=1.0, width=0.001, 
             headwidth=7, headlength=10,
             linewidth=0.5, edgecolor='r', color='k')
  #plt.plot(x_data, y_data, 'r-', linewidth = 1)
  plt.xlabel(X_VARIABLE_LABEL, fontsize = 16)
  plt.ylabel(Y_VARIABLE_LABEL, fontsize = 16)
  plt.title(test_name, fontsize = 16)

  # Legend text (material parameters)
  legend_text = yieldParams['legend']
  plt.subplots_adjust(right = 0.75)
  plt.figtext(0.77, 0.90, legend_text, ha='left', va='top', size='x-small')
  plt.grid()

  plt.savefig(pdf_name + ".pdf", bbox_inches = 'tight')
  #plt.show()
  plt.close(fig)

  return

def plot_extracted_stress(ups_file, data, stress_scale_fac):
  ''' '''
  print '\nPlotting extracted stress :\t'

  # Move to a smaller var name and get units
  fac = stress_scale_fac
  units = "(MPa)"
  if (fac < 1.0e-6):
    units = "(GPa)"

  # Get the yield surface parameters
  yieldParams = getMaterialParameters(ups_file)
  
  # Set plot defaults
  fontSize = 14
  lineWidth = 2
  plt.rcParams['font.size'] = fontSize
  plt.rcParams['font.weight'] = 'normal'
  plt.rcParams['lines.linewidth'] = lineWidth
  plt.rcParams['mathtext.fontset'] = 'stixsans'

  # Find the test name
  test_name = ups_file.split(".ups")[0]

  # Plot stress (pq) with yield surface
  plotExtractedStress(test_name, test_name + "pq", data,
                      data.sig_I1, "p", -fac, "$\\mathbf{p = I_1/3}$ "+units,
                      data.sig_J2, "q", -fac, "$\\mathbf{q = \\sqrt{3J_2}}$ "+units,
                      data.capX, data.alpha_I1, yieldParams)

  # Plot quasistatic stress (zr) with yield surface
  plotExtractedStress(test_name, test_name + "zr", data,
                      data.sigQS_I1, "z", -fac, "$\\mathbf{z = I_1/\\sqrt{3}}$ "+units,
                      data.sigQS_J2, "r", -fac, "$\\mathbf{r = \\sqrt{2J_2}}$ "+units,
                      data.capX, data.alpha_I1, yieldParams)

  print('Plotting extracted data done.')  

  return 

#---------------------------------------------------------------------------
# Plot the simulated vs. experimental stress
#---------------------------------------------------------------------------
def plotSimVsExptStressTime(test_name, pdf_name,
                          x_data, X_OPERATION, x_scale, X_VARIABLE_LABEL, 
                          y_data, Y_OPERATION, y_scale, Y_VARIABLE_LABEL,
                          expt_x_data, expt_x_scale,
                          expt_y_data, expt_y_scale):

  print "Variables: ", X_VARIABLE_LABEL, ", ", Y_VARIABLE_LABEL

  # Convert and scale data if needed
  x_data = map(lambda xx: convertAndScaleData(np.asarray(xx), X_OPERATION, x_scale), 
               x_data)
  y_data = map(lambda yy: convertAndScaleData(np.asarray(yy), Y_OPERATION, y_scale), 
               y_data)
  expt_x_data = map(lambda xx: convertAndScaleData(np.asarray(xx), X_OPERATION, expt_x_scale), 
               expt_x_data)
  expt_y_data = map(lambda yy: convertAndScaleData(np.asarray(yy), Y_OPERATION, expt_y_scale), 
               expt_y_data)

  # Plot the data
  #setPlotParams()
  #plt.rcdefaults()
  fig = plt.figure(figsize=(6,6))
  #ax1 = plt.subplot(111)
  x_data = np.array(x_data)
  y_data = np.array(y_data)
  expt_x_data = np.array(expt_x_data)
  expt_y_data = np.array(expt_y_data)
  plt.quiver(x_data[:-1], y_data[:-1], 
             x_data[1:]-x_data[:-1], y_data[1:]-y_data[:-1], 
             scale_units='xy', angles='xy', scale=1.0, width=0.001, 
             headwidth=7, headlength=10,
             linewidth=0.5, edgecolor='b', color='k')
  plt.plot(expt_x_data, expt_y_data, 'r--', label = "Experiment")
  plt.xlabel(X_VARIABLE_LABEL, fontsize = 16)
  plt.ylabel(Y_VARIABLE_LABEL, fontsize = 16)
  plt.title(test_name, fontsize = 16)
  plt.legend(loc='best', prop={'size':10})
  #savePDF(pdf_name)
  plt.savefig(pdf_name + ".pdf", bbox_inches = 'tight')
  #plt.show()
  plt.close(fig)

  return

def plotSimVsExptStressYieldSurf(test_name, pdf_name, sim_data, expt_data,
                        I1_data, X_OPERATION, x_scale, X_VARIABLE_LABEL, 
                        J2_data, Y_OPERATION, y_scale, Y_VARIABLE_LABEL,
                        capX_data, alpha_I1_data, yieldParams,
                        expt_I1_data, expt_J2_data, expt_scale): 

  print "Variables: ", X_VARIABLE_LABEL, ", ", Y_VARIABLE_LABEL

  # Convert and scale data if needed
  x_data = map(lambda xx: convertAndScaleData(np.asarray(xx), X_OPERATION, x_scale), 
               I1_data)
  y_data = map(lambda yy: convertAndScaleData(np.asarray(yy), Y_OPERATION, y_scale), 
               J2_data)
  x_expt_data = map(lambda xx: convertAndScaleData(np.asarray(xx), X_OPERATION, expt_scale), 
               expt_I1_data)
  y_expt_data = map(lambda yy: convertAndScaleData(np.asarray(yy), Y_OPERATION, expt_scale), 
               expt_J2_data)

  # Choose a paired colormap
  colors = map(lambda ii: cm.Paired(ii),
               np.linspace(0.0, 1.0, len(x_data)))

  # Create a figure
  fig = plt.figure(figsize=(8,6))
  res = map(lambda X, alpha, COLOR, time, PEAKI1, FSLOPE, STREN, YSLOPE : 
         computeAndPlotYieldSurface(yieldParams, time,
                                    PEAKI1, FSLOPE, STREN, YSLOPE,
                                    X, alpha, 
                                    X_OPERATION, x_scale,  
                                    Y_OPERATION, y_scale, COLOR, pdf_name),
         capX_data, alpha_I1_data, colors, sim_data.time,
         sim_data.PEAKI1, sim_data.FSLOPE, sim_data.STREN, sim_data.YSLOPE)

  #ax1 = plt.subplot(111)
  x_data = np.array(x_data)
  y_data = np.array(y_data)
  x_expt_data = np.array(x_expt_data)
  y_expt_data = np.array(y_expt_data)
  plt.quiver(x_data[:-1], y_data[:-1], 
             x_data[1:]-x_data[:-1], y_data[1:]-y_data[:-1], 
             scale_units='xy', angles='xy', scale=1.0, width=0.001, 
             headwidth=7, headlength=10,
             linewidth=0.5, edgecolor='r', color='k', label="Simulation")
  plt.plot(x_expt_data, y_expt_data, 'b--', linewidth = 1, label="Experiment")
  plt.xlabel(X_VARIABLE_LABEL, fontsize = 16)
  plt.ylabel(Y_VARIABLE_LABEL, fontsize = 16)
  plt.title(test_name, fontsize = 16)
  plt.legend(loc='best', prop={'size':10})
  #plt.gca().set_aspect('equal', adjustable='box')

  # Legend text (material parameters)
  legend_text = yieldParams['legend']
  plt.subplots_adjust(right = 0.75)
  plt.figtext(0.77, 0.90, legend_text, ha='left', va='top', size='x-small')
  plt.grid()

  plt.savefig(pdf_name + ".pdf", bbox_inches = 'tight')
  #plt.show()
  plt.close(fig)

  # Write the data
  outputfile = pdf_name + ".dat"
  exptLabel = "Experiment"
  simuLabel = "Simulation"
  with open(outputfile, "a") as outfile:
    writer = csv.writer(outfile)
    for time, x, y in zip(sim_data.time, x_data, y_data):
      writer.writerow([time, x, y, simuLabel])
    for time, x, y in zip(expt_data.time, x_expt_data, y_expt_data):
      writer.writerow([time, x, y, exptLabel])

  return

def plot_sim_vs_expt_stress(ups_file, sim_data, expt_data, stress_scale_fac, 
                            expt_scale_fac):
  ''' '''
  print '\nPlotting simulated vs. expt stress :\t'

  # Move to a smaller var name and get units
  fac = stress_scale_fac
  units = "(MPa)"
  if (fac < 1.0e-6):
    units = "(GPa)"

  # Get the yield surface parameters
  yieldParams = getMaterialParameters(ups_file)
  
  # Set plot defaults
  fontSize = 14
  lineWidth = 2
  plt.rcParams['font.size'] = fontSize
  plt.rcParams['font.weight'] = 'normal'
  plt.rcParams['lines.linewidth'] = lineWidth
  plt.rcParams['mathtext.fontset'] = 'stixsans'

  # Find the test name
  test_name = ups_file.split(".ups")[0]

  # Plot stress vs. strain
  plotSimVsExptStressTime(test_name, test_name + "_vs_expt_S11E11",
                        [F - 1.0 for F in sim_data.F_11], "none", -1.0, "$\\mathbf{\\epsilon_{11}}$", 
                        sim_data.sig_11, "none", -fac, "$\\mathbf{\\sigma_{11}}$ "+units,
                        expt_data.eps_11, 1.0, 
                        expt_data.sig_11, expt_scale_fac)

  # Plot time vs stress_11 
  plotSimVsExptStressTime(test_name, test_name + "_vs_expt_S11time",
                        sim_data.time, "none", 1.0e6, "$\\mathbf{t}$ " + "$\\mu$s", 
                        sim_data.sig_11, "none", -fac, "$\\mathbf{\\sigma_{11}}$ "+units,
                        expt_data.time, 1.0e6, 
                        expt_data.sig_11, expt_scale_fac)

  plotSimVsExptStressTime(test_name, test_name + "_vs_expt_S22time",
                        sim_data.time, "none", 1.0e6, "$\\mathbf{t}$ " + "$\\mu$s", 
                        sim_data.sig_22, "none", -fac, "$\\mathbf{\\sigma_{22}}$ "+units,
                        expt_data.time, 1.0e6, 
                        expt_data.sig_22, expt_scale_fac)

  plotSimVsExptStressTime(test_name, test_name + "_vs_expt_ptime",
                        sim_data.time, "none", 1.0e6, "$\\mathbf{t}$ " + "$\\mu$s", 
                        sim_data.sig_I1, "p", -fac, "$\\mathbf{p = I_1/3}$ "+units,
                        expt_data.time, 1.0e6, 
                        expt_data.sig_I1, expt_scale_fac)

  plotSimVsExptStressTime(test_name, test_name + "_vs_expt_qtime",
                        sim_data.time, "none", 1.0e6, "$\\mathbf{t}$ " + "$\\mu$s", 
                        sim_data.sig_J2, "q", -fac, "$\\mathbf{q = \\sqrt{3J_2}}$ "+units,
                        expt_data.time, 1.0e6, 
                        expt_data.sig_J2, expt_scale_fac)

  plotSimVsExptStressTime(test_name, test_name + "_vs_expt_ztime",
                        sim_data.time, "none", 1.0e6, "$\\mathbf{t}$ " + "$\\mu$s", 
                        sim_data.sig_I1, "z", -fac, "$\\mathbf{z = I_1/\\sqrt{3}}$ "+units,
                        expt_data.time, 1.0e6, 
                        expt_data.sig_I1, expt_scale_fac)

  plotSimVsExptStressTime(test_name, test_name + "_vs_expt_rtime",
                        sim_data.time, "none", 1.0e6, "$\\mathbf{t}$ " + "$\\mu$s", 
                        sim_data.sig_J2, "r", -fac, "$\\mathbf{r = \\sqrt{2J_2}}$ "+units,
                        expt_data.time, 1.0e6, 
                        expt_data.sig_J2, expt_scale_fac)

  # Plot stress (pq) with yield surface
  plotSimVsExptStressYieldSurf(test_name, test_name + "_vs_expt_pq", 
                      sim_data, expt_data,
                      sim_data.sig_I1, "p", -fac, "$\\mathbf{p = I_1/3}$ "+units,
                      sim_data.sig_J2, "q", -fac, "$\\mathbf{q = \\sqrt{3J_2}}$ "+units,
                      sim_data.capX, sim_data.alpha_I1, yieldParams,
                      expt_data.sig_I1, expt_data.sig_J2, expt_scale_fac)

  # Plot quasistatic stress (zr) with yield surface
  plotSimVsExptStressYieldSurf(test_name, test_name + "_vs_expt_zr", 
                      sim_data, expt_data,
                      sim_data.sigQS_I1, "z", -fac, "$\\mathbf{z = I_1/\\sqrt{3}}$ "+units,
                      sim_data.sigQS_J2, "r", -fac, "$\\mathbf{r = \\sqrt{2J_2}}$ "+units,
                      sim_data.capX, sim_data.alpha_I1, yieldParams,
                      expt_data.sig_I1, expt_data.sig_J2, expt_scale_fac)

  print('Plotting sim. vs. expt data done.')  

  return 

#---------------------------------------------------------------------------
# Extract data from all the tests
#---------------------------------------------------------------------------
def plot_all_tests():

  global root_dir, ARENA_PART_SAT_TEST_LIST

  print '#-- Extract data from all tests --#'
  for ups_file, expt_file in ARENA_PART_SAT_TEST_LIST:
    data = read_extracted_test_data(root_dir, ups_file)
    expt_data = read_expt_data(root_dir, expt_file)
    stress_scale_fac = 1.0e-6;
    expt_scale_fac = 1.0;
    if (abs(max(data.sig_I1)) > 1.0e9):
      stress_scale_fac = 1.0e-9;
      expt_scale_fac = 1.0e-3;
    plot_extracted_test_data(ups_file, data, stress_scale_fac)
    #plot_extracted_stress(ups_file, data, stress_scale_fac)
    plot_sim_vs_expt_stress(ups_file, data, expt_data, stress_scale_fac, expt_scale_fac)
    os.chdir(root_dir)
    
#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
if __name__ == "__main__":

  plot_all_tests()


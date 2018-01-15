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


# Assume that uda files are in this directory and links to executables are
# in the parent directory
root_dir = os.path.abspath(".")

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
# Plot two scalar particle variables from the extracted particle variable data
#---------------------------------------------------------------------------
def convertAndScaleData(data, OPERATION_TYPE, scale_fator):
  data = data.astype(np.float)
  result = {
    'none': lambda data: data, 
    'log': lambda data: np.log(data),
    'I1': lambda data: data,
    'sqrtJ2': lambda data: np.sign(data)*np.sqrt(abs(data)),
    'p': lambda data: data/3.0,
    'q': lambda data: np.sign(data)*np.sqrt(3.0*abs(data)),
    'z': lambda data: data/np.sqrt(3.0),
    'r': lambda data: np.sign(data)*np.sqrt(2.0*abs(data))
  }[OPERATION_TYPE](data)

  result = result*scale_fator

  return result

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

def computeAndPlotYieldSurface(PEAKI1, FSLOPE, STREN, YSLOPE, CR, 
                               capX, alpha_I1, 
                               X_OPERATION, x_scale, Y_OPERATION, y_scale, 
                               COLOR, LINEWIDTH):

  # Set up constants
  a1 = STREN
  a2 = (FSLOPE-YSLOPE)/(STREN-YSLOPE*PEAKI1)
  a3 = (STREN-YSLOPE*PEAKI1)*np.exp(-a2*PEAKI1)
  a4 = YSLOPE

  # Compute kappa
  X_eff = capX - alpha_I1
  kappa = PEAKI1 - CR*(PEAKI1 - X_eff)

  # Create an array of I1_eff values
  num_points = 100
  I1_eff_list = np.linspace((1.0 - 1.0e-6)*X_eff, PEAKI1, num_points)
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
  line1 = plt.plot(x_data,  y_data, '-b', linewidth = LINEWIDTH)
  line2 = plt.plot(x_data,  y_data_neg, '-b', linewidth = LINEWIDTH)
  plt.setp(line1, color=COLOR)
  plt.setp(line2, color=COLOR)

def plotYieldSurfaces(median_PEAKI1, median_FSLOPE, median_STREN, median_YSLOPE,
                      param_data, pdf_name, 
                      X_OPERATION, x_scale, X_VARIABLE_LABEL, 
                      Y_OPERATION, y_scale, Y_VARIABLE_LABEL,
                      capX_data, alpha_I1_data):

  print "Variables: ", X_VARIABLE_LABEL, ", ", Y_VARIABLE_LABEL
  print "median values = " , median_PEAKI1, median_FSLOPE, median_STREN, median_YSLOPE

  # Choose a paired colormap
  colors = map(lambda ii: cm.Paired(ii),
               np.linspace(0.0, 1.0, len(param_data.PEAKI1)))

  # Create a figure
  fig = plt.figure(figsize=(8,6))
  
  # Plot
  res = map(lambda X, alpha, COLOR, PEAKI1, FSLOPE, STREN, YSLOPE, CR : 
            computeAndPlotYieldSurface(PEAKI1, FSLOPE, STREN, YSLOPE, CR,
                                       X, alpha, 
                                       X_OPERATION, x_scale,  
                                       Y_OPERATION, y_scale, COLOR, 1),
            capX_data, alpha_I1_data, colors,
            param_data.PEAKI1, param_data.FSLOPE, param_data.STREN, param_data.YSLOPE,
            param_data.CR)

  # Plot median value
  computeAndPlotYieldSurface(median_PEAKI1, median_FSLOPE, median_STREN, median_YSLOPE, 
                              param_data.CR[0],
                              capX_data[0], alpha_I1_data[0], 
                              X_OPERATION, x_scale,  
                              Y_OPERATION, y_scale, (0.0, 0.0, 0.0, 1.0), 2)

  plt.xlabel(X_VARIABLE_LABEL, fontsize = 16)
  plt.ylabel(Y_VARIABLE_LABEL, fontsize = 16)
  plt.grid()

  plt.savefig(pdf_name + ".pdf", bbox_inches = 'tight')
  plt.show()
  plt.close(fig)

  return

def plot_yield_surface(median_PEAKI1, median_FSLOPE, median_STREN, median_YSLOPE,
                       param_data, pdf_name, stress_scale_fac, capX, alpha_I1):
  ''' '''
  print '\nPlotting yield surfaces :\t'

  # Move to a smaller var name and get units
  fac = stress_scale_fac
  units = "(Pa)"
  if (fac < 1.0):
    units = "(kPa)"
  if (fac < 1.0e-3):
    units = "(MPa)"
  if (fac < 1.0e-6):
    units = "(GPa)"

  # Set plot defaults
  fontSize = 14
  lineWidth = 2
  plt.rcParams['font.size'] = fontSize
  plt.rcParams['font.weight'] = 'normal'
  plt.rcParams['lines.linewidth'] = lineWidth
  plt.rcParams['mathtext.fontset'] = 'stixsans'

  # Plot stress (pq) with yield surface
  plotYieldSurfaces(median_PEAKI1, median_FSLOPE, median_STREN, median_YSLOPE,
                    param_data, pdf_name,
                    "I1", -fac, "$\\mathbf{I_1}$ "+units,
                    "sqrtJ2", -fac, "$\\mathbf{\\sqrt{J_2}}$ "+units,
                    capX, alpha_I1)

  # Plot stress (pq) with yield surface
  plotYieldSurfaces(median_PEAKI1, median_FSLOPE, median_STREN, median_YSLOPE,
                    param_data, pdf_name,
                    "p", -fac, "$\\mathbf{p = I_1/3}$ "+units,
                    "q", -fac, "$\\mathbf{q = \\sqrt{3J_2}}$ "+units,
                    capX, alpha_I1)

  print('Plotting yield surfaces done.')  

  return 

def computeWeibullValues(median_val, weibull_modulus, reference_vol, 
                         element_vol, seed, num_values):

  np.random.seed(seed)
  size_factor = (reference_vol/element_vol)**(1.0/weibull_modulus)
  weibull_scale = median_val/(np.log(2))**(1.0/weibull_modulus)
  weibull_val = np.random.weibull(weibull_modulus, num_values)
  sample_val = size_factor*weibull_scale*weibull_val

  return sample_val


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
if __name__ == "__main__":

  num_values = 100
  median_PEAKI1 = 1000
  modulus_PEAKI1 = 4
  median_FSLOPE = 0.453
  modulus_FSLOPE = 4
  median_YSLOPE = 0.31
  modulus_YSLOPE = 4 
  median_STREN = 1.0e7
  modulus_STREN = 4
  refvol = 0.001
  #refvol = 1.0e-6
  elemvol = 1.0e-6
  seed = 1

  values_PEAKI1 = computeWeibullValues(median_PEAKI1, modulus_PEAKI1, refvol, 
                                       elemvol, seed, num_values)
  values_FSLOPE = computeWeibullValues(median_FSLOPE, modulus_FSLOPE, refvol, 
                                       elemvol, 2*seed, num_values)
  values_STREN = computeWeibullValues(median_STREN, modulus_STREN, refvol, 
                                       elemvol, 3*seed, num_values)
  values_YSLOPE = computeWeibullValues(median_YSLOPE, modulus_YSLOPE, refvol, 
                                       elemvol, 4*seed, num_values)
  values_CR = np.full(num_values, 0.5)
  print(values_PEAKI1)
  print(values_FSLOPE)
  print(values_STREN)
  print(values_YSLOPE)

  # Save the data as a named tuple
  ReadData = collections.namedtuple('ReadData', ['PEAKI1', 'FSLOPE', 'STREN', 'YSLOPE', 'CR'])
  param_data = ReadData(values_PEAKI1, values_FSLOPE, values_STREN, values_YSLOPE, values_CR)

  # Scale the median data
  size_factor = (refvol/elemvol)**(1.0/modulus_PEAKI1)
  print("Size factor = ",size_factor)
  print "median values = " , median_PEAKI1, median_FSLOPE, median_STREN, median_YSLOPE
  median_PEAKI1 = size_factor*median_PEAKI1
  median_FSLOPE = size_factor*median_FSLOPE
  median_STREN = size_factor*median_STREN
  median_YSLOPE = size_factor*median_YSLOPE
  print "median values = " , median_PEAKI1, median_FSLOPE, median_STREN, median_YSLOPE

  # Set up CapX and alphaI1
  capX = -1.0e5
  values_capX = np.full(num_values, capX)
  alpha_I1 = 0.0
  values_alpha_I1 = np.full(num_values, alpha_I1)
  pdf_name = "test"
  stress_scale_fac = 1.0e-3;
  plot_yield_surface(median_PEAKI1, median_FSLOPE, median_STREN, median_YSLOPE,
                     param_data, pdf_name, stress_scale_fac, values_capX, values_alpha_I1)


#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

#Shorthand character combinations
tab = '\t'
endl = '\n'
One = '1.0'
Zed = '0.0'
dblZed = Zed+tab+Zed
rotation = Zed+tab+Zed+tab+Zed+tab+Zed+endl
firstLine = Zed+tab+One+tab+dblZed+tab+Zed+tab+One+tab+Zed+tab+dblZed+tab+One+tab+rotation

#output numerical precision
precision = '1.8e'

traceMonitor = []

#Write each leg to file
def linearInterpEpsil(F,tInit,delT,delEpsil_11,delEpsil_22,delEpsil_33,epsil11_init=0.0,epsil22_init=0.0,epsil33_init=0.0,numPnts = 100):
  epsilSlope_11 = delEpsil_11/delT
  epsilSlope_22 = delEpsil_22/delT
  epsilSlope_33 = delEpsil_33/delT
  subDelT = delT/numPnts
  time = tInit
  epsil_11 = epsil11_init
  epsil_22 = epsil22_init
  epsil_33 = epsil33_init
  for i in range(numPnts):
    time += subDelT
    epsil_11 += subDelT*epsilSlope_11
    epsil_22 += subDelT*epsilSlope_22
    epsil_33 += subDelT*epsilSlope_33
    F11 = np.exp(epsil_11)
    F22 = np.exp(epsil_22)
    F33 = np.exp(epsil_33)
    F.write(format(time,precision)+tab+format(F11,precision)+tab+dblZed+tab+Zed+tab+format(F22,precision)+tab+Zed+tab+dblZed+tab+format(F33,precision)+tab+rotation)

def write_Fline(FileHandle,t,F):
  out_str = format(t,precision)+tab
  for i in range(3):
    for j in range(3):
      out_str+=format(F[i][j],precision)+tab
  out_str+=rotation
  FileHandle.write(out_str)
  
def dyadicMultiply(a,b):
  T = np.eye(3)
  for i in range(3):
    for j in range(3):
      T[i][j] = a[i]*b[j]
  return T
  
def tensor_exp(A):
  #Get eigen values and associated eigenvectors
  eigVals,eigVecs = np.linalg.eig(A)
  #Find repeated roots conditions
  if eigVals[0] == eigVals[1] and eigVals[1] != eigVals[2]:
    OneTwoRepeated = True
    TwoThreeRepeated = False
    AllEqual = False
  elif eigVals[0] != eigVals[1] and eigVals[1] == eigVals[2]:
    OneTwoRepeated = False
    TwoThreeRepeated = True
    AllEqual = False      
  elif eigVals[0] == eigVals[1] and eigVals[0] == eigVals[2]:
    OneTwoRepeated = False
    TwoThreeRepeated = False
    AllEqual = True
  else:
    OneTwoRepeated = False
    TwoThreeRepeated = False
    AllEqual = False
  #Use spectral decomposition to compute the log of Epsilon to get F
  expA = np.zeros((3,3))  
  if OneTwoRepeated:
    expA += np.exp(eigVals[1])*(dyadicMultiply(eigVecs[0],eigVecs[0])+dyadicMultiply(eigVecs[1],eigVecs[1]))    
    expA += np.exp(eigVals[2])*dyadicMultiply(eigVecs[2],eigVecs[2])    
  elif TwoThreeRepeated:
    expA += np.exp(eigVals[0])*dyadicMultiply(eigVecs[0],eigVecs[0])
    expA += np.exp(eigVals[1])*(dyadicMultiply(eigVecs[1],eigVecs[1])+dyadicMultiply(eigVecs[2],eigVecs[2]))
  elif AllEqual:
    expA += np.exp(eigVals[0])*(dyadicMultiply(eigVecs[0],eigVecs[0])+dyadicMultiply(eigVecs[1],eigVecs[1])+dyadicMultiply(eigVecs[2],eigVecs[2])) 
  else:
    for i in range(3):
      expA += np.exp(eigVals[i])*dyadicMultiply(eigVecs[i],eigVecs[i])
  traceMonitor.append(expA.trace())
  return expA  
  
def elastic_leg(FileHandle,numPnts=100):
  Sqrt6 = np.sqrt(6.0)
  t = 0.0
  epsil_11 = 0.0
  epsil_22 = 0.0
  epsil_33 = 0.0  
  subDelT = 1.0/numPnts
  for i in range(numPnts-1):
    t += subDelT
    e11 = (-2.0*t)/(200.0*Sqrt6)
    e22 = t/(200.0*Sqrt6)
    e33 = t/(200.0*Sqrt6)
    Epsil = np.array([[e11,0,0],[0,e22,0],[0,0,e33]])
    F = tensor_exp(Epsil)
    write_Fline(FileHandle,t,F)

def plastic_leg(FileHandle,numPnts=100):
  Sqrt2 = np.sqrt(2.0)  
  Sqrt3 = np.sqrt(3.0) 
  epsil_11 = 0.0
  epsil_22 = 0.0
  epsil_33 = 0.0  
  subDelT = 4.0/numPnts
  t = 1.0
  for i in range(numPnts):
    t += subDelT
    e11 = ( 6.0*Sqrt2*np.cos(np.pi*t*0.5) -np.pi*(3.0 +4.0*Sqrt2 -3.0*t +Sqrt2*t +15.0*Sqrt2*np.sin(np.pi*t*0.5)))/(4000.0*Sqrt3*np.pi)
    e22 = (-6.0*Sqrt2*np.cos(np.pi*t*0.5) -np.pi*(3.0 +4.0*Sqrt2 -3.0*t +Sqrt2*t -15.0*Sqrt2*np.sin(np.pi*t*0.5)))/(4000.0*Sqrt3*np.pi)
    e33 = (-3.0 +8.0*Sqrt2 +(3.0 +2.0*Sqrt2)*t)/(4000.0*Sqrt3)
    e12 = (Sqrt3*(-2.0 +5.0*np.pi*np.cos(np.pi*t*0.5) +2.0*np.sin(np.pi*t*0.5)))/(2000.0*Sqrt2*np.pi)
    Epsil = np.array([[e11,e12,0],[e12,e22,0],[0,0,e33]])
    F = tensor_exp(Epsil)
    write_Fline(FileHandle,t,F)

F = open("AreniscaTest_12_PrescribedDeformation.inp","w")
F.write(firstLine)
elastic_leg(F,500)
plastic_leg(F,500)
F.close

#from matplotlib import pyplot as plt
#plt.plot(traceMonitor,marker='*')
#plt.show()

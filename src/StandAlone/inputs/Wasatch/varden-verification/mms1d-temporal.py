# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 09:20:07 2014

@author: tsaad
"""

import numpy as np
import argparse
import os
from xml.dom import minidom
import sys
from shutil import copyfile
from uintahcleanup import uintahcleanup
from upsparser import parseups
parser = argparse.ArgumentParser(description=
                                 'Process an uda to get TKE information. Note'+
                                 ' that this script assumes that: '+
                                 ' 1) you have a local copy of lineextract \n'+ 
                                 ' 2) you are timestepping at dt=1e-3 \n'+
                                 ' 3) you have the cbc_spectrum.txt data local ')

parser.add_argument('-ups',
                    help='The input file to run.')                    

parser.add_argument('-levels',
                    help='The number of time refinement levels.')                    
                    
args = parser.parse_args()

if args.levels is None:    
  print 'Error'
  sys.exit()
    
if args.ups is not None:
  rootups = args.ups
  nLevels =int(args.levels)
  dt0 = 0.001
  maxTimeSteps = 2
  parseups(rootups, nLevels,dt0,maxTimeSteps)

  #now load the data and compute the errors committed

  # starting with velocities
  uAll = []
  uErrAll = []  
  print '*************** OBSERVED TEMPORAL ORDER ***************'
  print '---------------- VELOCITY (u) -------------------'  
  for i in range(0,nLevels):
    datname = 'u-t' + str(i) + '.txt'
    u = np.loadtxt(datname)
    uAll.append(u[:,3])
  xxvol = u[:,0]
  tmp = []
  refinement = 1
  for i in range(0,nLevels):
    t = maxTimeSteps*dt0/refinement
    refinement *= 2
    A = 2.5
    uExact = -(2.0*2.5*t)/(t*t + 1)*np.sin(2.0*np.pi*xxvol/(3.0*t + 30.0))
    diff = uExact - uAll[i]
    uerr = np.linalg.norm(diff,2)
    tmp.append(uerr)
  for i in range(0,nLevels-1):
    ratio = tmp[i]/tmp[i+1]
    print np.log(ratio)/np.log(2.0)
    
  # now do the scalars
  fAll = []
  fErrAll = []  
  print '---------------- MIXTURE FRACTION (f) -------------------'
  for i in range(0,nLevels):
    datname = 'f-t' + str(i) + '.txt'
    f = np.loadtxt(datname)
    fAll.append(f[:,3])
  xsvol = f[:,0]

  tmp = []
  refinement = 1
  for i in range(0,nLevels):
    t = maxTimeSteps*dt0/refinement
    refinement *= 2
    fExact = (5.0/(2.0*t + 5.0)) * np.exp(-5.0*(xsvol*xsvol)/(10.0+t))   
    diff = fExact - fAll[i]
    ferr = np.linalg.norm(diff,2)
    tmp.append(ferr)
  for i in range(0,nLevels-1):
    ratio = tmp[i]/tmp[i+1]
    print np.log(ratio)/np.log(2.0)

  uintahcleanup()


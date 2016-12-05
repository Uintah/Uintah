# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:58:41 2015
@author: Tony Saad
"""
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import os
from xml.dom import minidom
from shutil import copyfile
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
"""
Given a 3D array A of size (Nx, Ny, Nz) (representative of a CFD mesh), 
this function computes a new array B of size (Nx/2, Ny/2, Nz/2)
such that the entries in B are the averaged values of corresponding cells in A.
Specifically, for a cell centered scalar quantity that lives on A, every cell
in B corresponds to the average of the 8 cells in A.
@author: Tony Saad
"""
def average(phi):
  # get the dimensions of the input array
  shape = phi.shape
  nx0 = shape[0]
  ny0 = shape[1]
  nz0 = shape[2]
  # we will average two points in each dimension
  nx = nx0/2
  ny = ny0/2
  nz = nz0/2
  phiAv = np.zeros([nx,ny,nz])
  for iav in range(0,nx):
    for jav in range(0,ny):
      for kav in range(0,nz):   
        i = 2*iav
        j = 2*jav
        k = 2*kav
        average = (phi[i,j,k] + phi[i+1,j,k] + phi[i,j+1,k] + phi[i,j,k+1] + phi[i+1,j+1,k] + phi[i+1,j,k+1] + phi[i,j+1,k+1] + phi[i+1,j+1,k+1])/8.0
#        average = (phi[i,j,k] + phi[i,j+1,k] + phi[i,j,k+1] + phi[i,j+1,k+1] )/4.0
        phiAv[iav,jav,kav] = average
  return phiAv

#------------------------------------------------------------------------------
def main():
  parser = argparse.ArgumentParser(description=
                                   'Computes spatial order of accuracy without the need of an anlytical solution. The method '+
                                    'is based on computing numerical solutions at refined timesteps and then computing the '+
                                    'order as p = ln[(f3 - f2)/(f2 - f1)]/ln(0.5).' +
                                    ' The cleanest way to operate this script is to make a copy of it in a new directory. Then '+
                                    'copy the ups file to that directory and execute the script.' )
  
  parser.add_argument('-ups',
                      help='The input file to run.',required=True)                    
  
  parser.add_argument('-levels',
                      help='The number of spatial refinement levels.', type=int)                    
  
  parser.add_argument('-nsteps',
                      help='The number of timesteps. Defaults to 1.', type=int)                    
  
  parser.add_argument('-suspath',
                      help='The path to sus.',required=True)
  
  parser.add_argument('-vars', required=True,
                      help='Comma seperated list of variables for which the temporal order is to be computed. example: -vars "var1, my var".')
                      
  args = parser.parse_args()
  
  # if the number of levels is not provided, set it to 3
  if args.levels is None:
    args.levels = 3
  
  # if the number of levels is <2, then reset it to 3
  if (args.levels < 2):
    print 'The number of levels has to be >= 3. Setting levels to 3'
    args.levels = 3
  
  rootups = args.ups
  nLevels = args.levels
  
  # cleanup the list of variables for which the order is to be computed
  myvars = [x.strip() for x in args.vars.split(',')]
  
  # first makes copies of the ups files
  fnames = []
  basename = os.path.basename(rootups)
  basename = os.path.splitext(basename)[0]
  for i in range(0,nLevels):
    #fname = os.path.splitext(rootups)[0] + '-t' + str(i) + '.ups'    
    fname = basename + '-t' + str(i) + '.ups'
    fnames.append(fname)
    copyfile(rootups, fname)    
    
  # now loop over the copied files and change the dt and the uda name
  refinement = 1
  maxSteps = 1
  
  if args.nsteps is not None:
    maxSteps = args.nsteps
  
  args.suspath = os.path.normpath(args.suspath)
  args.suspath = os.path.abspath(args.suspath)
  print args.suspath
  os.system('ln -fs ' + args.suspath + '/sus sus')
  os.system('ln -fs ' + args.suspath + '/tools/extractors/lineextract lineextract')
  
  # find total number of procs and resolution
  xmldoc = minidom.parse(rootups)
  for node in xmldoc.getElementsByTagName('patches'):
      P = (str(node.firstChild.data).strip()).split(',')
      P0=int(P[0].split('[')[1])
      P1=int(P[1])
      P2=int(P[2].split(']')[0])
  total_proc = P0*P1*P2
  
  for node in xmldoc.getElementsByTagName('resolution'):
      P = (str(node.firstChild.data).strip()).split(',')
      Nx=int(P[0].split('[')[1])
      Ny=int(P[1])
      Nz=int(P[2].split(']')[0])
  
  for fname in fnames:
    print 'now updating xml for ', fname
    basename = os.path.splitext(fname)[0]
    xmldoc = minidom.parse(fname)
  
    for node in xmldoc.getElementsByTagName('filebase'):
      node.firstChild.replaceWholeText(basename + '.uda')
  
    for node in xmldoc.getElementsByTagName('resolution'):
      node.firstChild.replaceWholeText('[' + str(Nx*refinement) + ',' + str(Ny*refinement) + ',' + str(Nz*refinement) + ']')
  
    for node in xmldoc.getElementsByTagName('max_Timesteps'):
      node.firstChild.replaceWholeText(maxSteps*refinement)
         
    for node in xmldoc.getElementsByTagName('delt_min'):
      dtmin = float(node.firstChild.data)
      dtmin = dtmin/refinement
      node.firstChild.replaceWholeText(dtmin)
  
    for node in xmldoc.getElementsByTagName('delt_max'):
      node.firstChild.replaceWholeText(dtmin)
  
    for node in xmldoc.getElementsByTagName('outputTimestepInterval'):
      node.firstChild.replaceWholeText('1')
  
    for node in xmldoc.getElementsByTagName('maxTime'):
      node.firstChild.replaceWholeText('100')
      
    refinement *= 2
    f = open(fname, 'w') 
    xmldoc.writexml(f) 
    f.close()
  
  # now run the files
  counter = 0
  refinement = 1
  for fname in fnames:
    os.system('mpirun -np '+ str(total_proc) + ' ' + './sus' + ' ' + fname + ' > log.txt')
    udaName = os.path.splitext(fname)[0] + '.uda'
    #    #EXTRACT THE variables
    for var in myvars:                      
      outFile = str(var) + '-t' + str(counter) + '.txt'
      the_command = './lineextract -pr 32 -q -v ' + str(var) + ' -timestep ' + str(maxSteps*refinement) + ' -istart 0 0 0 -iend ' + str(Nx*refinement - 1)+' '+str(Ny*refinement -1)+' '+str(Nz*refinement - 1)+ ' -o ' + outFile +' -uda '+udaName
      print 'Executing command: ', the_command
      os.system(the_command)
      
    os.system('rm ' + fname)    
    refinement *= 2
    counter += 1
  
  #now load the data and compute the errors
  print '---------------- SPATIAL ORDER -------------------'
  for var in myvars:    
    phiAll = []
    refinement = 1
    for i in range(0,nLevels):
      datname = str(var) + '-t' + str(i) + '.txt'
      phi = np.loadtxt(datname)
      phi = np.reshape(phi[:,3],(Nx*refinement,Ny*refinement,Nz*refinement),'F') # take the last column of phi and reshape
      phiAll.append(phi)
  #    phit = average(phi) # average phi
  #    plt.matshow(phi[:,:,0])
  #    plt.matshow(phit[:,:,0])
  #    plt.show()
      refinement *= 2
      os.system('rm ' + datname)
    
    # local errors
    errAll = []
    for i in range(0,nLevels-1):
      #phiav = average(phiAll[i+1])    
      diff = average(phiAll[i+1]) - phiAll[i]
      #plt.matshow(diff[:,:,0])    
      shape = diff.shape
      size = shape[0]*shape[1]*shape[2]
      diff = diff.reshape(size)
      err = np.linalg.norm(diff,np.inf)
      errAll.append(err)
  
    #plt.show()    
    # now compute order
    print '-----------------------------'    
    print ' VARIABLE: ', var
    print '-----------------------------'
    for i in range(0,nLevels-2):
      print np.log( errAll[i+1]/errAll[i] ) / np.log(0.5)
  
  os.system('rm -rf *.uda*')
  os.system('rm -rf *.dot')
  os.system('rm log.txt')  

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
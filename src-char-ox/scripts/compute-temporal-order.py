# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 11:03:18 2014
Last updated on Dec 29, 2014
@author: tsaad
"""

import numpy as np
import argparse
import os
from xml.dom import minidom
from shutil import copyfile

parser = argparse.ArgumentParser(description=
                                 'Computes temporal order of accuracy without the need of an anlytical solution. The method '+
                                  'is based on computing numerical solutions at refined timesteps and then computing the '+
                                  'order as p = ln[(f3 - f2)/(f2 - f1)]/ln(0.5).' +
                                  ' The cleanest way to operate this script is to make a copy of it in a new directory. Then '+
                                  'copy the ups file to that directory and execute the script.' )

parser.add_argument('-ups',
                    help='The input file to run.',required=True)                    

parser.add_argument('-levels',
                    help='The number of time refinement levels.', type=int)                    

parser.add_argument('-nsteps',
                    help='The number of timesteps. Defaults to 10.', type=int)                    

parser.add_argument('-dt',
                    help='The initial timestep size that is to be refined. Defaults to delt_min in the ups file.', type=float)                    

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
nLevels =args.levels

# cleanup the list of variables for which the order is to be computed
myvars = [x.strip() for x in args.vars.split(',')]

# first makes copies of the ups files
fnames = []
for i in range(0,nLevels):
  fname = os.path.splitext(rootups)[0] + '-t' + str(i) + '.ups'    
  fnames.append(fname)
  copyfile(rootups, fname)    
  
# now loop over the copied files and change the dt and the uda name
refinement = 1
maxSteps = 10

if args.nsteps is not None:
  maxSteps = args.nsteps

dt0 = 1e-3
dtspecified = False
if args.dt is not None:
  dtspecified = True
  dt0 = args.dt

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
    
  for node in xmldoc.getElementsByTagName('delt_min'):
    if dtspecified:
      dtmin = args.dt
    else:
      dtmin = float(node.firstChild.data)
    dtmin = dtmin/refinement
    node.firstChild.replaceWholeText(dtmin)

  for node in xmldoc.getElementsByTagName('delt_max'):
    #dtmax = float(node.firstChild.data)
    #dtmax = dtmax/refinement
    node.firstChild.replaceWholeText(dtmin)

  foundMaxTimesteps = False
  for node in xmldoc.getElementsByTagName('max_Timesteps'):
    foundMaxTimesteps = True
    node.firstChild.replaceWholeText(maxSteps*refinement)
  
  if (not foundMaxTimesteps):
    for node in xmldoc.getElementsByTagName('Time'):
      foundMaxTimesteps = True
      maxT = xmldoc.createElement('max_Timesteps')
      text = xmldoc.createTextNode(str(maxSteps*refinement))
      maxT.appendChild(text)
      node.appendChild(maxT)
   
  needsTimestepInterval = True
  for node in xmldoc.getElementsByTagName('outputInterval'):
    needsTimestepInterval = True
    parent = node.parentNode
    parent.removeChild(node)

  for node in xmldoc.getElementsByTagName('outputTimestepInterval'):
    needsTimestepInterval = False
    node.firstChild.replaceWholeText('1')    
    
  if (needsTimestepInterval):
    for node in xmldoc.getElementsByTagName('DataArchiver'):
      outInter = xmldoc.createElement('outputTimestepInterval')
      text = xmldoc.createTextNode('1')
      outInter.appendChild(text)
      node.appendChild(outInter)

  hasInitTimestep = False
  for node in xmldoc.getElementsByTagName('outputInitTimestep'):
    hasInitTimestep = True
  
  if (not hasInitTimestep):
    for node in xmldoc.getElementsByTagName('DataArchiver'):
      outInter = xmldoc.createElement('outputInitTimestep')
      node.appendChild(outInter)  
  
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
    the_command = './lineextract -pr 32 -v ' + str(var) + ' -timestep ' + str(maxSteps*refinement) + ' -istart 0 0 0 -iend ' + str(Nx-1)+' '+str(Ny-1)+' '+str(Nz-1)+ ' -o ' + outFile +' -uda '+udaName
    print 'Executing command: ', the_command
    os.system(the_command)
    
  os.system('rm ' + fname)    
  refinement *= 2
  counter += 1

#now load the data and compute the errors
print '---------------- TEMPORAL ORDER -------------------'
for var in myvars:    
  phiAll = []
  for i in range(0,nLevels):
    datname = str(var) + '-t' + str(i) + '.txt'
    phi = np.loadtxt(datname)
    phiAll.append(phi[:,3])
    os.system('rm ' + datname)

  print '-----------------------------'    
  print ' VARIABLE: ', var
  print '-----------------------------'

  # local errors
  errAll = []
  for i in range(0,nLevels-1):
    diff = phiAll[i+1] - phiAll[i]
    err = np.linalg.norm(diff,2)
    print 'error', err
    errAll.append(err)
    
  # now compute order
  for i in range(0,nLevels-2):
    print 'order:', np.log( errAll[i+1]/errAll[i] ) / np.log(0.5)

os.system('rm -rf *.uda*')
os.system('rm -rf *.dot')
os.system('rm log.txt')  

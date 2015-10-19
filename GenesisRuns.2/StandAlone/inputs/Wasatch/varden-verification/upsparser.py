# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 09:20:07 2014

@author: tsaad
"""
import os
from xml.dom import minidom
from shutil import copyfile

def parseups(rootups,nLevels,dt0,maxTimeSteps ):

  # first make copies of the ups files
  fnames = []
  for i in range(0,nLevels):
    fname = os.path.splitext(rootups)[0] + '-t' + str(i) + '.ups'    
    fnames.append(fname)
    copyfile(rootups, fname)    
    
    
  # now loop over the copied files and change the dt and the uda name
  refinement = 1

  # find total number of procs
  xmldoc = minidom.parse(rootups)
  for node in xmldoc.getElementsByTagName('patches'):
      P = (str(node.firstChild.data).strip()).split(',')
      P0=int(P[0].split('[')[1])
      P1=int(P[1])
      P2=int(P[2].split(']')[0])
  totalProc= P0*P1*P2  
  
  # find resolution
  for node in xmldoc.getElementsByTagName('resolution'):
      P = (str(node.firstChild.data).strip()).split(',')
      nx=int(P[0].split('[')[1])
      ny=int(P[1])
      nz=int(P[2].split(']')[0])
  
  # modify copied files  
  for fname in fnames:    
    basename = os.path.splitext(fname)[0]
    xmldoc = minidom.parse(fname)
    for node in xmldoc.getElementsByTagName('filebase'):
      node.firstChild.replaceWholeText(basename + '.uda')
    for node in xmldoc.getElementsByTagName('delt_min'):
      dtmin = dt0/refinement
      node.firstChild.replaceWholeText(dtmin)
    for node in xmldoc.getElementsByTagName('delt_max'):
      dtmax = dt0/refinement
      node.firstChild.replaceWholeText(dtmax)
    for node in xmldoc.getElementsByTagName('max_Timesteps'):
      node.firstChild.replaceWholeText(maxTimeSteps)
    for node in xmldoc.getElementsByTagName('outputTimestepInterval'):
      node.firstChild.replaceWholeText('1')
    for node in xmldoc.getElementsByTagName('maxTime'):
      node.firstChild.replaceWholeText('100')
      
    refinement *= 2
    #maxSteps *= 2
    f = open(fname, 'w') 
    xmldoc.writexml(f) 
    f.close()
  
  # now run the files
  counter = 0
  for fname in fnames:
    command = 'mpirun -np '+ str(totalProc) + ' ./sus ' + fname + ' > log.txt' 
    print 'now executing command: ', command
    os.system(command)
    udaName = os.path.splitext(fname)[0] + '.uda'
    # extract data                
    print 'extracting data...'
    outFile = 'f' + '-t' + str(counter) + '.txt'
    lineExtractCommand = './lineextract -v f -cellCoords -pr 32 -timestep ' + str(maxTimeSteps) + ' -istart 0 0 0 -iend ' + str(nx)+' '+str(ny)+' '+str(nz)+ ' -o ' + outFile +' -uda ' + udaName + ' > log.txt'              
    os.system(lineExtractCommand)
    
    outFile = 'u' + '-t' + str(counter) + '.txt'
    lineExtractCommand = './lineextract -v u -cellCoords -pr 32 -timestep ' + str(maxTimeSteps) + ' -istart 0 0 0 -iend ' + str(nx)+' '+str(ny)+' '+str(nz)+ ' -o ' + outFile +' -uda '+udaName + ' > log.txt' 
    os.system(lineExtractCommand)

#    outFile = 'rhof' + '-t' + str(counter) + '.txt'
#    lineExtractCommand = './lineextract -v rhof -cellCoords -timestep ' + str(maxTimeSteps) + ' -istart 0 0 0 -iend ' + str(P0)+' '+str(P1)+' '+str(P2)+ ' -o ' + outFile +' -uda '+udaName + ' > log.txt'         
#    os.system(lineExtractCommand)
#
#    outFile = 'x-mom' + '-t' + str(counter) + '.txt'
#    lineExtractCommand = './lineextract -v x-mom -cellCoords -timestep ' + str(maxTimeSteps) + ' -istart 0 0 0 -iend ' + str(P0)+' '+str(P1)+' '+str(P2)+ ' -o ' + outFile +' -uda '+udaName + ' > log.txt'          
#    os.system(lineExtractCommand)

    #lastTimestep *= 2
    counter += 1

  os.system('rm log.txt')
  return fnames,totalProc,nx,ny,nz
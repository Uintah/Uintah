# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:58:41 2015
@author: Tony Saad (modified by Oscar Diaz)
"""
# -*- coding: utf-8 -*-

#import numpy as np
import argparse
import os
from xml.dom import minidom
from shutil import copyfile
#import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
def main():
  parser = argparse.ArgumentParser(description='I need to write a description ...' )
  
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
                      
  parser.add_argument('-dire', required=True,
                      help='only for 1D verificartion')
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
 
  dire =  args.dire 
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
      #node.firstChild.replaceWholeText('[' + str(Nx*refinement) + ',' + str(Ny*refinement) + ',' + str(Nz*refinement) + ']')
      node.firstChild.replaceWholeText('[' + str(Nx) + ',' + str(Ny) + ',' + str(Nz) + ']')
  
    for node in xmldoc.getElementsByTagName('max_Timesteps'):
      node.firstChild.replaceWholeText(maxSteps*refinement)
      #node.firstChild.replaceWholeText(51*refinement)
         
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
      #node.firstChild.replaceWholeText('0.1')
      
    refinement *= 2
    f = open(fname, 'w') 
    xmldoc.writexml(f) 
    f.close()
  
  if dire =='x':
     fs = [0.5,.5,.5]
     fe = [0.5,.5,.5]
  elif dire == 'y':
     fs = [.5,0,.5]
     fe = [.5,1,.5]
  elif dire =='z':
     fs = [0.5,0.5,0]
     fe = [0.5,0.5,1]

  # now run the files
  counter = 0
  refinement = 1
  for fname in fnames:
    #os.system('mpirun -np '+ str(total_proc) + ' ' + './sus' +' -do_not_validate' + ' ' + fname + ' > log.txt')
    os.system('mpirun -np '+ str(total_proc) + ' ' + './sus' + ' ' + fname + ' > log.txt')
    udaName = os.path.splitext(fname)[0] + '.uda'
    p_s   = [int(fs[0]*Nx), int(fs[1]*Ny), int(fs[2]*Nz)] 
    p_end = [int(fe[0]*Nx), int(fe[1]*Ny), int(fe[2]*Nz)] 
    #    #EXTRACT THE variables
    for var in myvars:                      
      outFile = 'data/'+ str(var) + '-t' + str(counter) + '.txt'
      the_command = './lineextract -v ' + str(var) + ' -istart ' + str(p_s[0] )+' '+str(p_s[1])+' '+str(p_s[2])+' -iend ' + str(p_end[0] )+' '+str(p_end[1])+' '+str(p_end[2])+ ' -o ' + outFile +' -uda '+udaName
      print 'Executing command: ', the_command
      os.system(the_command)
      
    os.system('rm ' + fname)    
    refinement *= 2
    counter += 1

  os.system('rm -rf *.uda*')
  os.system('rm -rf *.dot')
  os.system('rm log.txt')  
  

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

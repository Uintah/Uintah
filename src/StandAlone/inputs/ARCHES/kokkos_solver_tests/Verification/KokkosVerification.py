# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:58:41 2015
@author: Oscar Diaz, Jeremy Thornock (originally based on Tony Saad's script but since heavily modified)
"""
# -*- coding: utf-8 -*-

import argparse
import os
from xml.dom import minidom
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import datetime
import subprocess

#------------------------------------------------------------------------------

def read_fileV2(name_file):
    f = np.loadtxt(name_file)
    x = f[:,0] # 
    y = f[:,1] # 
    z = f[:,2] # 
    Np = [0,0,0]
    Np[0] = int(np.max(x)) - int(np.min(x)) + 1 
    Np[1] = int(np.max(y)) - int(np.min(y)) + 1 
    Np[2] = int(np.max(z)) - int(np.min(z)) + 1 

    f = np.reshape(f[:,3],(Np[0],Np[1],Np[2]),'F') # take the last column of phi and reshape
    Nt = Np[0]*Np[1]*Np[2]
    return  f, Nt

def e(fe,f,dx,p =1.0):
    return (np.sum(abs(fe-f)**p)*dx)**(1./p)

def read_file(name_file):
    f  = np.loadtxt(name_file)
    Nt = len(f[:,0])
    L  = 1. 
    ts = np.linspace(0,L,Nt)
    return Nt, ts, f[:,-1]

def compute_temporal(data,var_name,var_mms,Nl,p):
    datname = [] 
    f = []
    fe = []
    fmms = []
    L1 = []
    for i in range(Nl):
        ## mms
        data_mms = data + '/' + var_mms + '-t'+str(i)+'.txt'
        Nt, ts, fm = read_file(data_mms)
        ## variable
        datname.append(data+ '/' + var_name + '-t'+str(i)+'.txt')
        Nt, ts, f0 = read_file(datname[i])

        f.append(f0)
        fe.append(fm)

        e0 = e(f0,fm,1./Nt,p = p)
        L1.append(e0)
    #plt.figure()
    #plt.plot(ts,f0,'o',label = var_mms)
    #plt.plot(ts,fm,'*r', label = var_name)
    #plt.xlabel('time [s]')
    #plt.ylabel('variable')
    #plt.legend(loc=2)
    #plt.savefig('temp_'+var_name+'RK')
    L1 = np.array(L1) 
    #plt.show()    
    #m, b, r_value, p_value, std_err = stats.linregress(np.log(dt),np.log(L1))
    #print 'm = ',m,'b = ', b, 'r_value = ' , r_value  
    #plt.loglog(dt,L1,'*--',label=var_name)
    return L1

def compute_spatial(data, var_name, var_mms, Nl, p):
    datname = [] 
    x =[]
    f = []
    fmms = []
    L1 = []
    for i in range(Nl):
        ## mms
        data_mms = data + '/' +var_mms + '-t'+str(i)+'.txt'
        fe, Nt = read_fileV2(data_mms) 
        ## variable
        datname.append(data +  '/' +var_name + '-t'+str(i)+'.txt')
        f0, Nt = read_fileV2(datname[i]) 
        f.append(f0)
        #e0 = e(f0,fe,DX[i],p = p)
        e0 = e(f0,fe,1./Nt,p = p)
        L1.append(e0)

    L1 = np.array(L1) 
    #dx = np.array(dx)
#    print x[0]
#    plt.figure()
#    plt.plot(x[0],mms(x[0], wmms),'*')
#    plt.plot(x0,fm0,'o')
#    plt.plot(x1,f1,'*')
#    plt.plot(x2,f2,'s')
#    plt.figure()
#    plt.plot(x0,abs(f0-fm0),'o')

    

#    plt.show()    
    return L1

def run_test(args):

  # if the number of levels is not provided, set it to 3
  if args.levels is None:
    args.levels = 3
  
  # if the number of levels is <2, then reset it to 3
  if (args.levels < 2):
    print('The number of levels has to be >= 3. Setting levels to 3')
    args.levels = 3
  
  rootups = args.ups
  nLevels = args.levels
  
  # cleanup the list of variables for which the order is to be computed
  myvars = [x.strip() for x in args.vars.split(',')]

  # first makes copies of the ups files
  fnames = []
  basename = os.path.basename(rootups)
  basename = os.path.splitext(basename)[0]
  start = 0 
  for i in range(start,nLevels+start):
    #fname = os.path.splitext(rootups)[0] + '-t' + str(i) + '.ups'    
    fname = basename + '-t' + str(i) + '.ups'
    fnames.append(fname)
    copyfile(rootups, fname)    
    
  # now loop over the copied files and change the dt and the uda name
  refinement = 1
  maxSteps = 1
  
  if args.nsteps is not None:
    maxSteps = args.nsteps

  if args.tstep is None:
    args.tstep = 0

  time_step = args.tstep 
  
  args.suspath = os.path.normpath(args.suspath)
  args.suspath = os.path.abspath(args.suspath)
  #print(args.suspath)
  sus = args.suspath + '/sus'
  lineextract =  args.suspath + '/tools/extractors/lineextract' 

  # axis for extraction 
  if args.axis is None:
    args.axis = 'x,y' 

  mydir = [x.strip() for x in args.axis.split(',')]
  
  if mydir[0] =='t':
    typeofanalysis = 'temporal'
  else:
    typeofanalysis = 'spatial'

  if args.bc is None:
    args.bc = 'none'
  mybc = [x.strip() for x in args.bc.split(',')]

  var_mms =  args.var_mms 
  myvars.append(var_mms) 

  # find total number of procs and resolution
  xmldoc = minidom.parse(rootups)
  for node in xmldoc.getElementsByTagName('patches'):
      P = (str(node.firstChild.data).strip()).split(',')
      P0=int(P[0].split('[')[1])
      P1=int(P[1])
      P2=int(P[2].split(']')[0])
  total_proc = P0*P1*P2
 
  for node in xmldoc.getElementsByTagName('lower'):
      P = (str(node.firstChild.data).strip()).split(',')
      L0=float(P[0].split('[')[1])
      L1=float(P[1])
      L2=float(P[2].split(']')[0])

  for node in xmldoc.getElementsByTagName('upper'):
      P = (str(node.firstChild.data).strip()).split(',')
      U0=float(P[0].split('[')[1])
      U1=float(P[1])
      U2=float(P[2].split(']')[0])

  for node in xmldoc.getElementsByTagName('resolution'):
      P = (str(node.firstChild.data).strip()).split(',')
      Nx=int(P[0].split('[')[1])
      Ny=int(P[1])
      Nz=int(P[2].split(']')[0])
 
  dt = [] 
  DX = [] 
  dx = [] 
  Lx = U0 - L0
  Ly = U1 - L1
  Lz = U2 - L2

  for fname in fnames:

    basename = os.path.splitext(fname)[0]
    xmldoc = minidom.parse(fname)

    for node in xmldoc.getElementsByTagName('filebase'):
      node.firstChild.replaceWholeText(basename + '.uda')
  
    for node in xmldoc.getElementsByTagName('delt_min'):
      dtmin = float(node.firstChild.data)

    for node in xmldoc.getElementsByTagName('maxTime'):
      maxTime = float(node.firstChild.data)

    if typeofanalysis == 'temporal' :

      for node in xmldoc.getElementsByTagName('max_Timesteps'):
        node.firstChild.replaceWholeText(maxSteps*refinement)
         
      for node in xmldoc.getElementsByTagName('delt_min'):
        dtmin = dtmin/refinement
        node.firstChild.replaceWholeText(dtmin)
  
    else :

      for node in xmldoc.getElementsByTagName('resolution'):
         node.firstChild.replaceWholeText('[' + str(Nx*refinement) + ',' + str(Ny*refinement) + ',' + str(Nz*refinement) + ']')
  
      dxyz = 1.
      d    = 0.
      count = 0.
      for dire in mydir:
        if dire == 'x':
          dxyz  *= (Lx/Nx/refinement)
          d     += (Lx/Nx/refinement) 
          count += 1.
        if dire == 'y':
          dxyz  *= (Ly/Ny/refinement)
          d     += (Ly/Ny/refinement) 
          count += 1.
        if dire == 'z':
          dxyz  *= (Lz/Nz/refinement)
          d     += (Lz/Nz/refinement) 
          count += 1.
      DX.append(dxyz)
      dx.append(d/count)

    if args.tsave is None:
      tsave = 1 
    else:
      tsave = int(float(args.tsave)/dtmin)

    for node in xmldoc.getElementsByTagName('outputTimestepInterval'):
      node.firstChild.replaceWholeText(tsave)

    # When I have a handoff plane 
    for node in xmldoc.getElementsByTagName('filename'):
      node.firstChild.replaceWholeText('scalars/2D/BC_mms/x_lr'+str(Nx*refinement)+'.dat')  

    for node in xmldoc.getElementsByTagName('delt_max'):
      node.firstChild.replaceWholeText(dtmin)
  
    dt.append(dtmin) 
    refinement *= 2
    f = open(fname, 'w') 
    xmldoc.writexml(f) 
    f.close()
  
  fs = [0.5,0.5,0.5]
  fe = [0.5,0.5,0.5]
  BCs = [0,0,0]
  BCe = [0,0,0]

  for dire in mydir:
    if dire == 'x':
      fs[0] = 0
      fe[0] = 1
    elif dire == 'y':
      fs[1] = 0
      fe[1] = 1
    elif dire == 'z':
      fs[2] = 0
      fe[2] = 1

  for dire in mybc:
    if dire == 'x':
      BCs[0] = -1 
      BCe[0] = 1 
    elif dire == 'y':
      BCs[1] = -1 
      BCe[1] = 1
    elif dire == 'z':
      BCs[2] = -1 
      BCe[2] = 1

  # directory to save data 
  data = 'Output_Verification_data_'+datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  subprocess.call('mkdir '+data, shell=True,executable='/bin/bash')

  # now run the files
  counter = 0
  refinement = 1

  for fname in fnames:

    command = 'mpirun -np '+ str(total_proc) + ' ' + sus  + ' ' + fname + ' >& log.txt'
    #print('running: '+command)
    subprocess.call(command,shell=True,executable='/bin/bash')
    udaName = os.path.splitext(fname)[0] + '.uda'
    p_s   = [int(fs[0]*Nx*refinement - BCs[0]), int(fs[1]*Ny*refinement - BCs[1]), int(fs[2]*Nz*refinement - BCs[2])] 
    p_end = [int(fe[0]*Nx*refinement - BCe[0]), int(fe[1]*Ny*refinement - BCe[1]), int(fe[2]*Nz*refinement - BCe[2])] 

    #EXTRACT THE variables
    for var in myvars:

      outFile = data + '/' + str(var) + '-t' + str(counter) + '.txt'

      if typeofanalysis == 'temporal' :
        the_command = lineextract + ' -v ' + str(var) + ' -istart ' + str(p_s[0] )+' '+str(p_s[1])+' '+str(p_s[2])+' -iend ' + str(p_end[0] )+' '+str(p_end[1])+' '+str(p_end[2])+ ' -o ' + outFile +' -uda '+udaName +' >& le.out'
      else:
        the_command = lineextract + ' -v ' + str(var) + ' -timestep '+ str(time_step) + ' -istart ' + str(p_s[0] )+' '+str(p_s[1])+' '+str(p_s[2])+' -iend ' + str(p_end[0] )+' '+str(p_end[1])+' '+str(p_end[2])+ ' -o ' + outFile +' -uda '+udaName +' >& le.out'

      #print('Running this command: '+the_command)
      subprocess.call(the_command,shell=True,executable='/bin/bash')

    subprocess.call('rm ' + fname, shell=True,executable='/bin/bash')    

    if typeofanalysis != 'temporal' :
      refinement *= 2

    counter += 1

  ### Here is where we compute m and b   #####
  Nl = nLevels 
  p = 2 # 
  convergence_results = {}

  for i,var in enumerate(myvars):
    if var !=var_mms:

       if typeofanalysis != 'temporal' :
         L1 = compute_spatial(data,var,var_mms,Nl,p)
         label_x = '$\Delta$ [m]'
       else:
         L1 = compute_temporal(data,var,var_mms,Nl,p)
         dx = np.copy(dt)
         label_x = '$\Delta$ [s]'

       m, b, r_value, p_value, std_err = stats.linregress(np.log(dx),np.log(L1))
       #print('m = '+np.str(m)+' b = '+np.str(b)+  ' r_value = ' +np.str(r_value)  )

       result = {'m':m, 'b':b, 'r':r_value}
       convergence_results[var] = result

       plt.figure()
       plt.loglog(dx,L1,'*--',label=var)
       plt.xlabel(label_x)
       plt.xlim([dx[0]*1.1,dx[-1]*0.9])
       plt.ylabel('|E| L= '+str(p))
       plt.legend(loc=3)
       plt.savefig(data+'/'+basename)

  #plt.show()
  if args.keep_uda is None:
    subprocess.call('rm -rf *.uda*',shell=True,executable='/bin/bash')
  subprocess.call('rm -rf *.dot',shell=True,executable='/bin/bash')
  subprocess.call('rm log.txt',shell=True,executable='/bin/bash')  

  return convergence_results

#------------------------------------------------------------------------------

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='I need to write a description ...' )
  
  parser.add_argument('-ups',
                      help='The input file to run.',required=True)                    
  
  parser.add_argument('-levels',
                      help='The number of spatial refinement levels.', type=int)                    
  
  parser.add_argument('-nsteps',
                      help='The number of timesteps. Defaults to 1.', type=int)                    
  
  parser.add_argument('-tstep',
                      help='The number of timesteps. Defaults to 1.', type=int)                    

  parser.add_argument('-tsave',
                      help='save time')                    

  parser.add_argument('-suspath',
                      help='The path to sus.',required=True)
  
  parser.add_argument('-vars', required=True,
                      help='Comma seperated list of variables for which the temporal order is to be computed. example: -vars "var1, my var".')
                      
  parser.add_argument('-axis',
                      help='axis where to extract data')

  parser.add_argument('-bc',
                      help='axis where there is a BC')

  parser.add_argument('-var_mms', required=True,
                      help='name of mms')

  parser.add_argument('-keep_uda', 
                      help='Keep the udas - do not delete them.', action='store_true')


  args = parser.parse_args()

  convergence_results = run_test(args)

  print(convergence_results)


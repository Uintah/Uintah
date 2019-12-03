# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 16:02:42 2014

@author: jeremy

"""

import numpy as np
import argparse
import os
from xml.dom import minidom
import matplotlib.pyplot as plt
import sys
import datetime

def tke_spectrum(u,v,w,L):
    #return the TKE spectum given: 
    #u,v,w** = the velocity field on 3D (i,j,k)
    # ** assumed as numpy arrays
    #L = length of the domain
    #returns a list with [wavenumber, energy]
    nt = np.size(u)
    uhat = np.fft.fftn(u)/nt
    vhat = np.fft.fftn(v)/nt
    what = np.fft.fftn(w)/nt
    
    n = u.shape[0]
    tkehat = np.zeros([n,n,n])
    
    tkehat = 0.5*(uhat*np.conjugate(uhat)+vhat*np.conjugate(vhat)+
    what*np.conjugate(what)).real
    
    #The largest waveform that can fit on L
    k0 = 2.0*np.pi/L
    #The max number of waves allowed for n grid points
    kmax = n/2.
    
    result = np.zeros([n+1,2])
    result[:,0] = np.linspace(0,n,num=n+1)*k0
    
    for kx in range(n):
        rkx = kx
        if ( kx > kmax ): rkx = rkx-n
        for ky in range(n):     
            rky = ky
            if ( rky > kmax ): rky = rky-n    
            for kz in range(n):         
                rkz = kz
                if ( rkz > kmax ): rkz = rkz-n
                k = int(np.round((rkx**2.+rky**2.+rkz**2.)**0.5,0))
                result[k,1] = result[k,1] + tkehat[kx,ky,kz]/k0                   
                    
    return result     

#----------------------------------#    
#------------- do it --------------#
#----------------------------------#
    
parser = argparse.ArgumentParser(description=
                                 'Process an uda to get TKE information. Note'+
                                 ' that this script assumes that: '+
                                 ' 1) you have a local copy of lineextract \n'+ 
                                 ' 2) you are timestepping at dt=1e-3 \n'+
                                 ' 3) you have the cbc_spectrum.txt data local ')
parser.add_argument('-uda', 
                    help='The uda to process. Assumes you have already'+
                    ' run the code. Only use this option if you dont specify'+
                    ' -sus and -ups.')
parser.add_argument('-sus',
                    help='The sus exe. [optional]')
parser.add_argument('-ups',
                    help='The input file to run. [needed if -sus is specified]')                    
parser.add_argument('-cleanup',help='Performs an rm -rf'+
                    ' *.uda* with user confirmation.', 
                    action='store_true')
parser.add_argument('-hardcleanup',help='USE WITH CAUTION: Performs an rm -rf'+
                    ' *.uda* - DELETES ALL UDAs in the directory(no prompt).', 
                    action='store_true')                    
parser.add_argument('-vel_base_name',help='Allows the user to specify a [base name] such '+
                   'that the velocities will be *[base name].')
                    
args = parser.parse_args()

if args.hardcleanup: 
    if args.uda is not None: 
        print('Resetting your cleanup flag because you specified -uda')
        args.hardcleanup = False

if args.cleanup: 
    if args.uda is not None: 
        print('Resetting your cleanup flag because you specified -uda')
        args.hardcleanup = False        

if args.uda is None:
    if args.ups is None:
        print('You need to specify the uda OR the ups file to run.')
        sys.exit()
        
if args.cleanup: 
    v = raw_input(['Really delete all UDAs in this directory?[y/n]'])
    if ( v == "y" or v == "yes"): 
        os.system('rm -rf *.uda*')
        
if args.hardcleanup: 
    os.system('rm -rf *.uda*')
    

    
if args.ups is not None: 
    xmldoc = minidom.parse(args.ups)
    for node in xmldoc.getElementsByTagName('patches'):
        P = (str(node.firstChild.data).strip()).split(',')
        P0=int(P[0].split('[')[1])
        P1=int(P[1])
        P2=int(P[2].split(']')[0])
    total_proc = P0*P1*P2
    os.system('mpirun -np '+str(total_proc)+' '+args.sus+' -mpi '+args.ups)

if args.uda is not None:                       
    xmldoc = minidom.parse(args.uda+'/input.xml')   

for node in xmldoc.getElementsByTagName('resolution'):
    P = (str(node.firstChild.data).strip()).split(',')
    P0=int(P[0].split('[')[1])
    P1=int(P[1])
    P2=int(P[2].split(']')[0])
    
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
    
for node in xmldoc.getElementsByTagName('filebase'):
    uda_name = str(node.firstChild.data)

for node in xmldoc.getElementsByTagName('delt_min'):
    dt_min = str(node.firstChild.data)

for node in xmldoc.getElementsByTagName('delt_max'):
    dt_max = str(node.firstChild.data)

if float(dt_min) != float(dt_max):
  print('DT min and DT max in the UPS must be equal and be either 1e-2 or 1e-3.')
  sys.exit()

print('Going to query UDA: '+ uda_name)
    
L = U0 - L0
#L = 2.0*np.pi*9.0/100.0

print('DT = '+np.str(dt_min))
if float(dt_min) < 5.0e-3:
  # compatible with a dt=0.001 timestep and output every 2 steps. 
  TS = [0,28,66]
else: 
  # compatible with a dt=0.01 timestep and output every 10 steps. 
  TS = [0,14,33]

cbc = np.loadtxt('cbc_spectrum.txt')

count = 1

this_dir = 'TKE_data_'+datetime.datetime.now().strftime("%y%m%d_%H%M%S")
os.system('mkdir '+this_dir)

velbase = 'VelocitySPBC'

if args.vel_base_name is not None:
    velbase = args.vel_base_name

for element in TS: 

    #EXTRACT THE VELOCITIES                
    the_command = './lineextract -v u'+velbase+'  -timestep '+str(element)+' -istart 0 0 0 -iend '+str(P0)+' '+str(P1)+' '+str(P2)+' -o uvelTKE.'+str(count)+' -uda '+uda_name              
    os.system(the_command)
    the_command = './lineextract -v v'+velbase+'  -timestep '+str(element)+' -istart 0 0 0 -iend '+str(P0)+' '+str(P1)+' '+str(P2)+' -o vvelTKE.'+str(count)+' -uda '+uda_name              
    os.system(the_command)
    the_command = './lineextract -v w'+velbase+'  -timestep '+str(element)+' -istart 0 0 0 -iend '+str(P0)+' '+str(P1)+' '+str(P2)+' -o wvelTKE.'+str(count)+' -uda '+uda_name              
    os.system(the_command)
    
    os.system('mv uvelTKE.* '+this_dir+'/.')
    os.system('mv vvelTKE.* '+this_dir+'/.')
    os.system('mv wvelTKE.* '+this_dir+'/.')
    
    u = np.loadtxt(this_dir+'/uvelTKE.'+str(count))[:,3].reshape([P0,P1,P2])
    v = np.loadtxt(this_dir+'/vvelTKE.'+str(count))[:,3].reshape([P0,P1,P2])
    w = np.loadtxt(this_dir+'/wvelTKE.'+str(count))[:,3].reshape([P0,P1,P2])

    r = tke_spectrum(u,v,w,L)

    plt.loglog(r[1:P0-1,0],r[1:P0-1,1],'o-')
    plt.loglog(cbc[:,0]*100.,cbc[:,count]*1.0e-6,'k--')
    plt.xlim([10, 1e3])
    plt.ylim([1e-6, 1e-3])
    k0 = 2.0*np.pi/L;
    k_nyquist = k0*P0/2.0;
    plt.loglog([k_nyquist,k_nyquist],[1e-6, 1e-3],'k--')
    #plt.show()
    count += 1    


print('All velocity results have been put in directory: '+this_dir)

plt.xlabel(r'k, 1/$\it{m}$')
plt.ylabel(r'E, $m^3/s2$')    
plt.show()

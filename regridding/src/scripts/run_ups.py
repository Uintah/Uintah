# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:07:29 2014

@author: jeremy
"""

import time
import argparse
import os
from xml.dom import minidom
import numpy as np

parser = argparse.ArgumentParser(description=
                                 'This script will run all UPS file in the '+
                                 'directory passed as an argument.')
parser.add_argument('-dir', 
                    help='Directory for the UPS files.', 
                    required=True)

parser.add_argument('-sus',help='Path to sus.',required=True)

parser.add_argument('-dovalidation','--dovalidation',action='store_true',
                    help='Do validation only.')
                    
parser.add_argument('-ignore',help='UPS to ignore', nargs='?', required=False)

parser.add_argument('-maxpatch',help='Max patch size allowed', required=False)

args = parser.parse_args()

UPS = os.listdir(args.dir)

stime=time.time()
print 'Starting tests...'

doval = False
if args.dovalidation is True: 
    doval = True
    
failed=list()
success=list()
ignored_files=[]

for myfile in UPS:
    if myfile.endswith('.ups'):
        
        #print '\n'
        #print 'Working on file: '+myfile
        
        if doval: 
            
            os.system(args.sus+
            ' -validate '+ args.dir+'/'+myfile+' >& RSout.'+myfile)
            
            os.system('tail -n 2 RSout.'+myfile+' > RSresult.'+myfile)
            checkfile = open('RSresult.'+myfile,'r')
            counter=-1
            for line in checkfile:
                counter += 1
                if ( counter == 0):
                    if (line[0:3]=='Val'):
                        #print myfile+' validated successfully!'
                        os.system('rm -rf RSresult.'+myfile)
                        os.system('rm -rf RSout.'+myfile)
                        success.append(myfile)
                    else: 
                        failed.append(myfile)
            
        else: 

            #get the patch layout 
            xmldoc = minidom.parse(args.dir+'/'+myfile)
            items = xmldoc.getElementsByTagName('patches')
            totpatch = 0
            for node in xmldoc.getElementsByTagName('patches'):
                P = (str(node.firstChild.data).strip()).split(',')
                P0=int(P[0].split('[')[1])
                P1=int(P[1])
                P2=int(P[2].split(']')[0])
                totpatch = P0*P1*P2
                print 'Total patches =',totpatch,' for test: ',myfile
                
            ignore_file = False
            if args.ignore is not None:
                for element in args.ignore: 
                    if myfile == str(element): 
                        ignore_file = True
                        ignored_files.append(myfile)
                        
            if args.maxpatch is not None:
                if totpatch > int(args.maxpatch): 
                    ignore_file = True
                    ignored_files.append(myfile)

            if (ignore_file == False):
                
                #print 'running file: ', myfile
                    
                os.system('mpirun -np '+str(totpatch) + ' '+args.sus+
                ' -mpi '+ args.dir+'/'+myfile+' >& RSout.'+myfile)

                os.system('tail -n 1 RSout.'+myfile+' > RSresult.'+myfile)
                checkfile = open('RSresult.'+myfile,'r')

                for line in checkfile: 
                    if (line[0:3]=='Sus'):
                        #print myfile+' completed successfully!'
                        os.system('rm -rf RSresult.'+myfile)
                        os.system('rm -rf RSout.'+myfile)
                        success.append(myfile)
                    else: 
                        failed.append(myfile)
                    
print 'Total time for all tests: ',time.time()-stime,' [sec]'    

if len(failed) > 0: 
    print '\n !!! The following tests failed !!!'
    for element in failed: 
        print element
    
if len(success) > 0:     
    print '\n +++ The following tests succeeded +++'
    for element in success:
        print element

if len(ignored_files) > 0: 
    print '\n --- The following tests were ignored ---'
    for element in ignored_files: 
        print element

print '\n'

#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import multiprocessing
import subprocess as sub_proc  

from AreniscaTestSuite_PostProc import *

#Automated script for running and post processing Arenisca verification tetsts. 
# Requires that ../opt/StandAlone/ and ../opt/StandAlone/tools/extractors/ be on working PATH
# If you would like to run tests with more than one processor than mpirun must also be on PATH
# defaults to max number of processors or number of patches whichever is lower

#List for tests requiring restart with damping off and new end time
#		ups end_time
RESTART_LIST = [[],[]]

#Post processing list, ie tests that have a method to do their post processing
POST_PROCESS_LIST = [
  'AreniscaTest_01_UniaxialStrainRotate.ups',
  'AreniscaTest_02_VertexTreatment.ups',
  'AreniscaTest_03a_UniaxialStrain_NoHardening.ups',
  'AreniscaTest_03b_UniaxialStrain_wIsotropicHardening.ups',
  'AreniscaTest_03c_UniaxialStrain_wKinematicHardening.ups',
  'AreniscaTest_04_CurvedYieldSurface.ups',
  'AreniscaTest_05_HydrostaticCompressionFixedCap.ups',
  'AreniscaTest_06_UniaxialStrainCapEvolution.ups',
  'AreniscaTest_07_HydrostaticCompressionCapEvolution.ups',
  'AreniscaTest_08_LoadingUnloading.ups',
  'AreniscaTest_09_FluidFilledPoreSpace.ups',
  'AreniscaTest_10_PureIsochoricStrainRates.ups',
  'AreniscaTest_11_UniaxialStrainJ2plasticity.ups',
]

#get uintah/src path as enviornmental variable
uintah_src_path = os.path.abspath(os.environ['UINTAH_SRC'])

#construct default paths based on location of uintah_src_path
default_inputs_path = uintah_src_path+'/StandAlone/inputs/MPM/Arenisca'
default_working_dir = uintah_src_path+'/StandAlone/inputs/MPM/Arenisca/test_run'
#If the working directory does not exist then make it.
if not os.path.exists(default_working_dir):
  os.makedirs(default_working_dir)
  
#Make plots directory
default_plot_dir = default_working_dir+'/Plots'
if not os.path.exists(default_plot_dir):
  os.makedirs(default_plot_dir)

#Build list of tets in the default inputs directory
TEST_LIST = []
tmp_test_list = os.listdir(default_inputs_path)
for item in tmp_test_list:
  if os.path.isfile(item):
    if '.ups' in item:
      TEST_LIST.append(os.path.abspath(item))
TEST_LIST.sort()

for test in TEST_LIST:
  print test

### COMMENT ME OUT!!!!!!! ###
TEST_LIST = [
  #TEST_LIST[0],	#Test 01
  #TEST_LIST[1],	#Test 02
  #TEST_LIST[2],	#Test 03a
  #TEST_LIST[3],	#Test 03b
  #TEST_LIST[4],	#Test 03c
  #TEST_LIST[5],	#Test 04
  #TEST_LIST[6],	#Test 05
  #TEST_LIST[7],	#Test 06
  #TEST_LIST[8],	#Test 07
  TEST_LIST[9],	#Test 08
  #TEST_LIST[10],	#Test 09
  #TEST_LIST[11],	#Test 10
  #TEST_LIST[12],  	#Test 11
  
  ]
### --------------------- ###

def copy_test_to(from_file,to_file):
  '''Reads in test ups file at from_file and writes to to_file while replacing
absolute references to prescribed deformation files with relative ones. Also
copys deformation file to same root folder.'''
  #Read in the from file and close
  F_from_file = open(from_file,"r")
  from_file_lines = F_from_file.read().split('\n')
  F_from_file.close()
  #Open/create the to file
  F_to_file = open(to_file,"w")
  to_file_root = os.path.split(to_file)[0]
  #Copy the ups but change the Prescribed def filebase also copy this file
  for line in from_file_lines:
    if '<PrescribedDeformationFile>' in line and '</PrescribedDeformationFile>' in line:
      def_file = line.split('<PrescribedDeformationFile>')[1].split('</PrescribedDeformationFile>')[0].strip()
      line = line.replace(def_file,def_file.split('inputs/MPM/Arenisca/')[1])
      def_file = def_file.split('inputs/MPM/Arenisca/')[1]
      shutil.copyfile(default_inputs_path+'/'+def_file,to_file_root+'/'+def_file)
    F_to_file.write(line+'\n')
  F_to_file.close()
  return os.path.abspath(to_file)
  
def setup_restart(uda_path,new_end_time):
  #Fix input file
  input_file = uda_path+'/input.xml'
  F = open(input_file,"r+")
  all_lines = F.read().split('\n')
  F.seek(0)
  for line in all_lines:
    if '<maxTime>' in line:
      line = '    <maxTime>'+format(new_end_time,'1.6e')+'</maxTime>'
    F.write(line+'\n')
  F.close()
  
  #Find checkpoint dirs and change damping
  for item in os.listdir(uda_path+"/checkpoints/"):
    tmp_file = uda_path+"/checkpoints/"+item+"/timestep.xml"
    if os.path.isfile(tmp_file):
      F = open(tmp_file,"r+")
      all_lines = F.read().split('\n')
      F.seek(0)
      for line in all_lines:
	if '<artificial_damping_coeff>' in line:
	  line = '    <artificial_damping_coeff>0.0</artificial_damping_coeff>'
	F.write(line+'\n')
      F.close()  

def run_test(ups_path,WITH_MPI=False,NUM_PROCS=1,RESTART=False,DAMPING_OFF_NEW_END_TIME=False):
  ''' '''
  print '\nRuning test:\t',os.path.split(ups_path)[1]
  #Determine uda path
  F_ups = open(ups_path,"r")
  ups_lines = F_ups.read()
  uda_path = './'+ups_lines.split('<filebase>')[1].split('</filebase>')[0].strip()
  F_ups.close()    
  
  #Determine root path
  root_path = os.path.split(os.path.abspath(ups_path))[0]
  #Change current working directory to root path
  os.chdir(root_path)
  #Open runlog
  F_log = open(root_path+'/TEST_RUNLOG_'+os.path.split(ups_path)[1],"w")
  #Construct the argument list for subprocess to use.
  if not(WITH_MPI) or int(NUM_PROCS)<=1:
    args = ['sus',os.path.split(ups_path)[1]]
  else:
    args = ['mpirun','-np',str(int(NUM_PROCS)),'sus','-mpi',os.path.split(ups_path)[1]]
  #Run the test and wait for it to complete
  tmp = sub_proc.Popen(args,stdout=F_log,stderr=sub_proc.PIPE)
  dummy = tmp.wait()
  F_log.close()
  #If test calls for retstart
  if RESTART:
    #If turn damping off and run to new end time
    if DAMPING_OFF_NEW_END_TIME:
      #Setup the restart by setting damping to zero and modifying end time
      print 'Setting <artificial_damping_coeff> to zero and restarting with new end time of ',format(NEW_END_TIME,'1.4e')
      setup_restart(uda_path,DAMPING_OFF_NEW_END_TIME)
      print 'Done.\nRestarting...'
      #Open new runlog
      F_log = open(root_path+'/TEST_RUNLOG_RESTART_'+os.split(ups_path)[1],"w")
      #Construct the argument list
      if not(WITH_MPI) or NUM_PROCS<=1:
	args = ['sus','-restart','-move',uda_path+'.000']
      else:
        args = ['mpirun','-np',str(int(NUM_PROCS)),'sus','-mpi','-restart','-move',uda_path+'.000']
      #Run the test and wait for it to complete
      tmp = sub_proc.Popen(args,stdout=F_log,stderr=sub_proc.PIPE)
      dummy = tmp.wait()
      F_log.close()
      uda_path = uda_path+'.001'
  else:
    uda_path = uda_path+'.000'

  print('Test done.')  
  return uda_path

def clear_uda(uda_path):
  print 'Deleting uda...'
  tmp = sub_proc.Popen(['rm','-rf',uda_path],stdout=sub_proc.PIPE,stderr=sub_proc.PIPE)
  dummy = tmp.wait()
  tmp = sub_proc.Popen(['rm',uda_path.split('.uda')[0]+'.uda'],stdout=sub_proc.PIPE,stderr=sub_proc.PIPE)
  dummy = tmp.wait()  
  print 'Done'

def run_all_tests(TEST_METHODS=False):
  global default_working_dir,TEST_LIST,RESTART_LIST,MPI_FLAG,NUM_CPUS,POST_PROCESS_LIST
  print '#-- Running All Tests --#'
  for test in TEST_LIST:
    #Copy to working directory
    ups_path=copy_test_to(test,default_working_dir+'/'+os.path.split(test)[1])
    #Run
    if test not in RESTART_LIST[0]:
      uda_path = run_test(ups_path,WITH_MPI=MPI_FLAG,NUM_PROCS=NUM_CPUS)
    else:
      new_end_time = RESTART_LIST[1][RESTART_LIST[0].index(test)]
      uda_path = run_test(ups_path,WITH_MPI=MPI_FLAG,NUM_PROCS=NUM_CPUS,DAMPING_OFF_NEW_END_TIME=new_end_time)
    #Post process if called for
    if TEST_METHODS:
      test_yield_surface(uda_path)
    else:
      post_proc(test,uda_path,default_plot_dir)
    #Clean up the uda
    clear_uda(uda_path)
    
    
def post_proc(test,uda_path,save_path):
  global POST_PROCESS_LIST
  test_name = os.path.split(test)[1]
  if test_name in POST_PROCESS_LIST:
    if test_name == 'AreniscaTest_01_UniaxialStrainRotate.ups':
      test01_postProc(uda_path,save_path)
    if test_name == 'AreniscaTest_02_VertexTreatment.ups':
      test02_postProc(uda_path,save_path)
    if test_name == 'AreniscaTest_03a_UniaxialStrain_NoHardening.ups':
      test03_postProc(uda_path,save_path,a=True)
    if test_name == 'AreniscaTest_03b_UniaxialStrain_wIsotropicHardening.ups':
      test03_postProc(uda_path,save_path,b=True)
    if test_name == 'AreniscaTest_03c_UniaxialStrain_wKinematicHardening.ups':
      test03_postProc(uda_path,save_path,c=True)
    if test_name == 'AreniscaTest_04_CurvedYieldSurface.ups':
      test04_postProc(uda_path,save_path)
    if test_name == 'AreniscaTest_05_HydrostaticCompressionFixedCap.ups':
      test05_postProc(uda_path,save_path)
    if test_name == 'AreniscaTest_06_UniaxialStrainCapEvolution.ups':
      test06_postProc(uda_path,save_path)
    if test_name == 'AreniscaTest_07_HydrostaticCompressionCapEvolution.ups':
      test07_postProc(uda_path,save_path)      
    if test_name == 'AreniscaTest_08_LoadingUnloading.ups':
      test08_postProc(uda_path,save_path)
    if test_name == 'AreniscaTest_09_FluidFilledPoreSpace.ups':
      test09_postProc(uda_path,save_path)
    if test_name == 'AreniscaTest_10_PureIsochoricStrainRates.ups':
      test10_postProc(uda_path,save_path,WORKING_PATH=default_working_dir)       
    if test_name == 'AreniscaTest_11_UniaxialStrainJ2plasticity.ups':
      test11_postProc(uda_path,save_path,WORKING_PATH=default_working_dir)       
  else:
    print '\nERROR: test: ',test,'\n\tNot on post processing list.\n'


if __name__ == "__main__":
  #Determine number of CPU cores
  NUM_CPUS = multiprocessing.cpu_count()
  #Set MPI Flag intially to false
  MPI_FLAG = False  

  #Not setup yet. Need to scan ups to determine # patches
  if False:
    ABORT = False  
    if len(sys.argv) ==3:
      if sys.argcv[1] == '-mpirun':
	MPI_FLAG = True
	try:
	  NUM_CPUS = int(sys.argv[2])
	except:
	  NUM_CPUS = 1
	  print '\nError: invalid number of processors entered with -mpirun flag'
      else:
	print '\nInvalid Arguments entered!\n\tuse: AreniscaTestSuite.py -mpirun <# processor cores>'
	ABORT = True
    else:
      not_done = True
      while not_done:
	mpi_check = raw_input("Would you like to run using mpirun? (Y/N/(A)bort)\n").lower()
	if mpi_check == 'y':
	  not_done_2 = True
	  MPI_FLAG = True
	  while not_done_2:
	    try:
	      num_cores = int(raw_input("Enter the number of cores to run on:\n"))
	      NUM_CPUS = num_cores
	      not_done_2 = False
	      not_done = False
	    except:
	      print 'Invalid entry. Please try again.'
	elif mpi_check == 'n':
	  not_done = False
	elif mpi_check == 'a':
	  not_done = False
	  ABORT = True
      if not(ABORT):
	if multiprocessing.cpu_count()<NUM_CPUS:
	  NUM_CPUS=multiprocessing.cpu_count()
	  print '\nWarning: number of cores requested more than are available locally.\n\t# cores set to: ',NUM_CPUS
	print ' '
	run_all_tests()      
  
  TEST_METHODS = False
  run_all_tests(TEST_METHODS)


#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import multiprocessing
import shlex
import subprocess as sub_proc  
import time

# List of tests
ARENA_PART_SAT_TEST_LIST = [
   'MasonSandUniaxialStrainSHPB_26.ups',
   'MasonSandUniaxialStrainSHPB-081612-003.ups',
   'MasonSandUniaxialStrainSHPBwDamage_26.ups',
   'MasonSandUniaxialStrainSHPBwDisagg_26.ups',
   'BoulderClayDryUniaxialStrainSHPB_Mean.ups',
   'BoulderClayUniaxialStrainSHPB-072313-001.ups',
   'MasonSandSHPB052212-001.ups',
   'MasonSandSHPB052212-006.ups',
   'MasonSandSHPB052212-011.ups',
   'MasonSandSHPB052212-016.ups',
   'BoulderClaySHPB072213-014.ups',
   'BoulderClaySHPB072313-013.ups',
   'BoulderClayDrySHPB-Erik.ups',
   'BoulderClayDrySHPB-oldK-Erik.ups',
   'BoulderClaySHPB072313-001.ups',
]

### COMMENT ME OUT!!!!!!! ###
ARENA_PART_SAT_TEST_LIST = [
  ARENA_PART_SAT_TEST_LIST[0],  # SHPB: sample 26
  ARENA_PART_SAT_TEST_LIST[1],  # SHPB 18 Wet: sample 3
  ARENA_PART_SAT_TEST_LIST[2],  # SHPB: sample 26 with damage
  ARENA_PART_SAT_TEST_LIST[3],  # SHPB: sample 26 with disaggregation
  ARENA_PART_SAT_TEST_LIST[4],  # SHPB: Boulder clay: Dry Mean
  ARENA_PART_SAT_TEST_LIST[5],  # SHPB: Boulder clay: Fully saturated
  ARENA_PART_SAT_TEST_LIST[6],  # SHPB: Mason sand: 1770 kg/m^3
  ARENA_PART_SAT_TEST_LIST[7],  # SHPB: Mason sand: 1700 kg/m^3
  ARENA_PART_SAT_TEST_LIST[8],  # SHPB: Mason sand: 1580 kg/m^3
  ARENA_PART_SAT_TEST_LIST[9],  # SHPB: Mason sand: 1640 kg/m^3
  ARENA_PART_SAT_TEST_LIST[10],  # SHPB: Boulder clay: 12.8% w/w
  ARENA_PART_SAT_TEST_LIST[11],  # SHPB: Boulder clay: 40.8% w/w
  ARENA_PART_SAT_TEST_LIST[12],  # SHPB: Boulder clay: Dry Mean Erik's parameters
  ARENA_PART_SAT_TEST_LIST[13],  # SHPB: Boulder clay: Dry Mean Erik's parameters
  ARENA_PART_SAT_TEST_LIST[14],  # SHPB: Boulder clay: 1300 kg/m^3: 40.8% w/w
  ]
### --------------------- ###

# Assume that uda files are in this directory and links to executables are
# in the parent directory
root_dir = os.path.abspath(".")
exe_dir = os.path.abspath("../..")

# Create directories for the test cases
for test_name in ARENA_PART_SAT_TEST_LIST:
  output_dir_name = os.path.join(root_dir, test_name.split(".ups")[0])
  if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)

# Get the executable names
selectpart_exe           = os.path.abspath(os.path.join(exe_dir, "selectpart"))
partextract_exe          = os.path.abspath(os.path.join(exe_dir, "partextract"))
posextract_exe           = os.path.abspath(os.path.join(exe_dir, "extractPos"))
posvelmassvolextract_exe = os.path.abspath(os.path.join(exe_dir, "extractPosVelMassVol"))
pscalarextract_exe       = os.path.abspath(os.path.join(exe_dir, "extractPscalar"))
pvectorextract_exe       = os.path.abspath(os.path.join(exe_dir, "extractPvec"))
pmatrixextract_exe       = os.path.abspath(os.path.join(exe_dir, "extractPmat"))
stressextract_exe        = os.path.abspath(os.path.join(exe_dir, "extractS"))
velextract_exe           = os.path.abspath(os.path.join(exe_dir, "extractV"))
lineextract_exe          = os.path.abspath(os.path.join(exe_dir, "lineextract"))

# Print the tests to be extracted
for test in ARENA_PART_SAT_TEST_LIST:
  print(test)

#---------------------------------------------------------------------------
# Select the particles for which data are to be extracted
#---------------------------------------------------------------------------
def selectParticles(uda_dir, output_dir, test_name):
  ''' '''
  partList_file_name = os.path.join(output_dir, test_name + ".particles")
  partList_file_id = open(partList_file_name, 'w')
  print(partList_file_name)
  command = (selectpart_exe + " -mat 0 -box 0 0 0 1 0 0 0 1 0 0 0 1 -timesteplow 0 -timestephigh 0 -uda " + 
             uda_dir)
  print(command)
  args = shlex.split(command)
  print(args)
  try:
    run_status = sub_proc.Popen(args, bufsize = -1, stdout = partList_file_id, stderr = sub_proc.PIPE)
    partList_file_id.close()
    output, error = run_status.communicate()
    if output:
      print "ret> ", run_status.returncode, " (std::output = ", output, ")"
    if error:
      print "ret> ", run_status.returncode, " (std::error = ", error.strip(), ")"
  except OSError as e:
    print "OSError > ", e.errno
    print "OSError > ", e.strerror
    print "OSError > ", e.filename
  except:
    print "Error > ", sys.exc_info()[0]

  return partList_file_name

#---------------------------------------------------------------------------
# Extract a particle variable from the selected particles
#---------------------------------------------------------------------------
def getExtractorExecutable(VARIABLE_TYPE):
  return {
    'scalar': pscalarextract_exe,
    'vector': pvectorextract_exe,
    'matrix': pmatrixextract_exe
  }[VARIABLE_TYPE]

def extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                   particle_variable, VARIABLE_TYPE):

  # Extract the particle variable name
  var_name = particle_variable.split('p.')[1]

  # Create a particle variable output file
  output_file_name = os.path.join(output_dir, var_name+".out")
  stdout_file_name = os.path.join(output_dir, var_name+".stdout")
  stderr_file_name = os.path.join(output_dir, var_name+".stderr")

  # Construct the command line
  extractor_exe = getExtractorExecutable(VARIABLE_TYPE)
  print(extractor_exe)
  command = (extractor_exe + ' -m 0 -partvar ' + particle_variable + ' -p ' + 
             partList_file_name + ' -uda ' + uda_dir + ' -o ' + output_file_name)
  print command
  #os.system(command)

  args = shlex.split(command)
  print args
  stdout_file_id = open(stdout_file_name, 'w')
  stderr_file_id = open(stderr_file_name, 'w')
  try:
    #proc = sub_proc.call(args)
    proc = sub_proc.Popen(args, bufsize = -1, stdout = stdout_file_id, stderr = stderr_file_id)
    stdout_file_id.close()
    stderr_file_id.close()
    time.sleep(2)
    proc.kill()
    #proc = sub_proc.Popen(command, shell = True, stdout = sub_proc.PIPE, stderr = sub_proc.PIPE)
    #while proc.poll() is None:
    #  output, error = proc.communicate()
    #  print("Still working...")
  except OSError as e:
    print "OSError > ", e.errno
    print "OSError > ", e.strerror
    print "OSError > ", e.filename
  except:
    #print "Error > ", sys.exc_info()[0]
    print "Error > ", sys.exc_info()

  return

#---------------------------------------------------------------------------
# Actually extract the data from a test
#---------------------------------------------------------------------------
def extract_test_data(root_dir, ups_file):
  ''' '''
  print '\nExtracting test data from :\t', ups_file

  # Find the test name
  test_name = ups_file.split(".ups")[0]
  print test_name

  # Find the uda name
  uda_dir = os.path.join(root_dir, test_name + ".uda")
  print uda_dir

  # Find the output directory name
  output_dir = os.path.join(root_dir, test_name)
  print output_dir

  # Change current working directory to the output directory
  os.chdir(output_dir)

  # Select the particles for which data are to be extracted
  partList_file_name = selectParticles(uda_dir, output_dir, test_name)

  # Create a deformation gradient output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.deformationMeasure", "matrix")

  # Create a stress output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.stress", "matrix")

  # Create a quasistatic stress output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.stressQS", "matrix")

  # Create a capX output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.capX", "scalar")

  # Create a plastic strain output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.plasticStrain", "matrix")

  # Create an elastic volumetric strain output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.elasticVolStrain", "scalar")

  # Create a plastic eq strain output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.plasticCumEqStrain", "scalar")

  # Create a plastic volumetric strain output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.plasticVolStrain", "scalar")

  # Create a pore pressure output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.porePressure", "matrix")

  # Create a porosity output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.porosity", "scalar")

  # Create a saturation output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.saturation", "scalar")

  # Create a coherence output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.COHER", "scalar")

  # Create a TGROW output file
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.TGROW", "scalar")

  # Create a yield parameter output files
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.ArenaPEAKI1", "scalar")
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.ArenaFSLOPE", "scalar")
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.ArenaSTREN", "scalar")
  extractPartVar(uda_dir, output_dir, test_name, partList_file_name,
                 "p.ArenaYSLOPE", "scalar")
  print('Extraction done.')  
  os.chdir(root_dir)

  return 

#---------------------------------------------------------------------------
# Extract data from all the tests
#---------------------------------------------------------------------------
def extract_all_tests():

  global root_dir, ARENA_PART_SAT_TEST_LIST

  print '#-- Extract data from all tests --#'
  for ups_file in ARENA_PART_SAT_TEST_LIST:
    extract_test_data(root_dir, ups_file)
    
#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
if __name__ == "__main__":

  extract_all_tests()

